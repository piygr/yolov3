import torch
import torch.nn as nn
import pytorch_lightning as pl

from loss import YoloLoss
import config as cfg

""" 
Information about architecture config:
Tuple is structured by (filters, kernel_size, stride) 
Every conv is a same convolution. 
List is structured by "B" indicating a residual block followed by the number of repeats
"S" is for scale prediction block and computing the yolo loss
"U" is for upsampling the feature map and concatenating with a previous layer
"""
config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8],
    (1024, 3, 2),
    ["B", 4],  # To this point is Darknet-53
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
]


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn_act, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act

    def forward(self, x):
        if self.use_bn_act:
            return self.leaky(self.bn(self.conv(x)))
        else:
            return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super().__init__()
        self.layers = nn.ModuleList()
        for repeat in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    CNNBlock(channels, channels // 2, kernel_size=1),
                    CNNBlock(channels // 2, channels, kernel_size=3, padding=1),
                )
            ]

        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x):
        for layer in self.layers:
            if self.use_residual:
                x = x + layer(x)
            else:
                x = layer(x)

        return x


class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pred = nn.Sequential(
            CNNBlock(in_channels, 2 * in_channels, kernel_size=3, padding=1),
            CNNBlock(
                2 * in_channels, (num_classes + 5) * 3, bn_act=False, kernel_size=1
            ),
        )
        self.num_classes = num_classes

    def forward(self, x):
        return (
            self.pred(x)
            .reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3])
            .permute(0, 1, 3, 4, 2)
        )


class YOLOv3LightningModel(pl.LightningModule):
    def __init__(self, in_channels=3, num_classes=20, anchors=None, S=None):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.layers = self._create_conv_layers()
        self.anchor_list = (
                torch.tensor(anchors)
                * torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
        )

        self.criterion = YoloLoss()

        self.metric = dict(
            total_train_steps=0,
            epoch_train_loss=[],
            epoch_train_acc=[],
            epoch_train_steps=0,
            total_val_steps=0,
            epoch_val_loss=[],
            epoch_val_acc=[],
            epoch_val_steps=0,
            train_loss=[],
            val_loss=[],
            train_acc=[],
            val_acc=[]
        )

    def forward(self, x):
        outputs = []  # for each scale
        route_connections = []
        for layer in self.layers:
            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x))
                continue

            x = layer(x)

            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                route_connections.append(x)

            elif isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()

        return outputs

    def _create_conv_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels

        for module in config:
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module
                layers.append(
                    CNNBlock(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=1 if kernel_size == 3 else 0,
                    )
                )
                in_channels = out_channels

            elif isinstance(module, list):
                num_repeats = module[1]
                layers.append(ResidualBlock(in_channels, num_repeats=num_repeats,))

            elif isinstance(module, str):
                if module == "S":
                    layers += [
                        ResidualBlock(in_channels, use_residual=False, num_repeats=1),
                        CNNBlock(in_channels, in_channels // 2, kernel_size=1),
                        ScalePrediction(in_channels // 2, num_classes=self.num_classes),
                    ]
                    in_channels = in_channels // 2

                elif module == "U":
                    layers.append(nn.Upsample(scale_factor=2),)
                    in_channels = in_channels * 3

        return layers


    def get_layer(self, idx):
        if idx < len(self.layers) and idx >= 0:
            return self.layers[idx]



    def training_step(self, train_batch, batch_idx):
        x, target = train_batch
        output = self.forward(x)
        loss = self.criterion(output, target, loss_dict=True, anchor_list=self.anchor_list)
        acc = self.criterion.check_class_accuracy(output, target, cfg.CONF_THRESHOLD)

        self.metric['total_train_steps'] += 1
        self.metric['epoch_train_steps'] += 1
        self.metric['epoch_train_loss'].append(loss)
        self.metric['epoch_train_acc'].append(acc)

        self.log_dict({'train_loss': loss['total_loss']})

        return loss['total_loss']


    def validation_step(self, val_batch, batch_idx):
        x, target = val_batch
        output = self.forward(x)
        loss = self.criterion(output, target, loss_dict=True, anchor_list=self.anchor_list)
        acc = self.criterion.check_class_accuracy(output, target, cfg.CONF_THRESHOLD)

        self.metric['total_val_steps'] += 1
        self.metric['epoch_val_steps'] += 1
        self.metric['epoch_val_loss'].append(loss)
        self.metric['epoch_val_acc'].append(acc)

        self.log_dict({'val_loss': loss['total_loss']})


    def on_validation_epoch_end(self):
        if self.metric['total_train_steps']:
            print('Epoch ', self.current_epoch)
            epoch_loss = 0
            epoch_acc = dict(
                correct_class=0,
                correct_noobj=0,
                correct_obj=0,
                total_class_preds=0,
                total_noobj=0,
                total_obj=0
            )
            for i in range(self.metric['epoch_train_steps']):
                lo = self.metric['epoch_train_loss'][i]
                epoch_loss += lo['total_loss']
                acc = self.metric['epoch_train_acc'][i]
                epoch_acc['correct_class'] += acc['correct_class']
                epoch_acc['correct_noobj'] += acc['correct_noobj']
                epoch_acc['correct_obj'] += acc['correct_obj']
                epoch_acc['total_class_preds'] += acc['total_class_preds']
                epoch_acc['total_noobj'] += acc['total_noobj']
                epoch_acc['total_obj'] += acc['total_obj']


            print("Train -")
            print(f"Class accuracy is: {(epoch_acc['correct_class']/(epoch_acc['total_class_preds']+1e-16))*100:2f}%")
            print(f"No obj accuracy is: {(epoch_acc['correct_noobj']/(epoch_acc['total_noobj']+1e-16))*100:2f}%")
            print(f"Obj accuracy is: {(epoch_acc['correct_obj']/(epoch_acc['total_obj']+1e-16))*100:2f}%")
            print(f"Total loss: {(epoch_loss/(len(self.metric['epoch_train_loss'])+1e-16)):2f}")

            self.metric['epoch_train_loss'] = []
            self.metric['epoch_train_acc'] = []
            self.metric['epoch_train_steps'] = 0

            #---
            epoch_loss = 0
            epoch_acc = dict(
                correct_class=0,
                correct_noobj=0,
                correct_obj=0,
                total_class_preds=0,
                total_noobj=0,
                total_obj=0
            )
            for i in range(self.metric['epoch_val_steps']):
                lo = self.metric['epoch_val_loss'][i]
                epoch_loss += lo['total_loss']
                acc = self.metric['epoch_val_acc'][i]
                epoch_acc['correct_class'] += acc['correct_class']
                epoch_acc['correct_noobj'] += acc['correct_noobj']
                epoch_acc['correct_obj'] += acc['correct_obj']
                epoch_acc['total_class_preds'] += acc['total_class_preds']
                epoch_acc['total_noobj'] += acc['total_noobj']
                epoch_acc['total_obj'] += acc['total_obj']

            print("Validation -")
            print(f"Class accuracy is: {(epoch_acc['correct_class']/(epoch_acc['total_class_preds']+1e-16))*100:2f}%")
            print(f"No obj accuracy is: {(epoch_acc['correct_noobj']/(epoch_acc['total_noobj']+1e-16))*100:2f}%")
            print(f"Obj accuracy is: {(epoch_acc['correct_obj']/(epoch_acc['total_obj']+1e-16))*100:2f}%")
            print(f"Total loss: {(epoch_loss/(len(self.metric['epoch_val_loss'])+1e-16)):2f}")

            self.metric['epoch_val_loss'] = []
            self.metric['epoch_val_acc'] = []
            self.metric['epoch_val_steps'] = 0

            print("Creating checkpoint...")
            self.trainer.save_checkpoint(cfg.CHECKPOINT_FILE)


    def test_step(self, test_batch, batch_idx):
        self.validation_step(test_batch, batch_idx)

    def train_dataloader(self):
        if not self.trainer.train_dataloader:
            self.trainer.fit_loop.setup_data()

        return self.trainer.train_dataloader

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                  max_lr=cfg.LEARNING_RATE,
                                                  epochs=self.trainer.max_epochs,
                                                  steps_per_epoch=len(self.train_dataloader()),
                                                  pct_start=8 / self.trainer.max_epochs,
                                                  div_factor=100,
                                                  final_div_factor=100,
                                                  three_phase=False,
                                                  verbose=False
                                                  )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                'interval': 'step', # or 'epoch'
                'frequency': 1
            },
        }


def sanity_check(model):
    x = torch.randn((2, 3, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE))
    out = model(x)
    assert model(x)[0].shape == (2, 3, cfg.IMAGE_SIZE // 32, cfg.IMAGE_SIZE // 32, cfg.NUM_CLASSES + 5)
    assert model(x)[1].shape == (2, 3, cfg.IMAGE_SIZE // 16, cfg.IMAGE_SIZE // 16, cfg.NUM_CLASSES + 5)
    assert model(x)[2].shape == (2, 3, cfg.IMAGE_SIZE // 8, cfg.IMAGE_SIZE // 8, cfg.NUM_CLASSES + 5)
    print("Success!")