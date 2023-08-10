from dataset import *
from models.YoloV3Lightning import *
import utils

def init(model, basic_sanity_check=True, find_max_lr=True, train=True, **kwargs):
    if basic_sanity_check:
        validate_dataset()
        sanity_check(model)
        print("Set basic_sanity_check to False to proceed")
    else:
        if find_max_lr:
            optimizer = kwargs.get('optimizer')
            criterion = kwargs.get('criterion')
            train_loader = kwargs.get('train_loader')
            utils.find_lr(model, optimizer, criterion, train_loader)
            print("Set find_max_lr to False to proceed further")
        else:

            train_loader = kwargs.get('train_loader')
            val_loader = kwargs.get('test_loader')

            if train:
                trainer = pl.Trainer(
                    precision=16,
                    max_epochs=cfg.NUM_EPOCHS,
                    accelerator='gpu'
                )

                cargs = {}
                if cfg.LOAD_MODEL:
                    cargs = dict(ckpt_path=cfg.CHECKPOINT_FILE)

                trainer.fit(model, train_loader, val_loader, **cargs)
            else:
                ckpt_file = kwargs.get('ckpt_file')
                if ckpt_file:
                    checkpoint = utils.load_model_from_checkpoint(cfg.DEVICE, file_name=ckpt_file)
                    model.load_state_dict(checkpoint['model'], strict=False)

            #-- Printing samples
            model.to(cfg.DEVICE)
            model.eval()
            cfg.IMG_DIR = cfg.DATASET + "/images/"
            cfg.LABEL_DIR = cfg.DATASET + "/labels/"
            eval_dataset = YOLODataset(
                cfg.DATASET + "/test.csv",
                transform=cfg.test_transforms,
                S=[cfg.IMAGE_SIZE // 32, cfg.IMAGE_SIZE // 16, cfg.IMAGE_SIZE // 8],
                img_dir=cfg.IMG_DIR,
                label_dir=cfg.LABEL_DIR,
                anchors=cfg.ANCHORS,
                mosaic=False
            )
            eval_loader = DataLoader(
                dataset=eval_dataset,
                batch_size=cfg.BATCH_SIZE,
                num_workers=cfg.NUM_WORKERS,
                pin_memory=cfg.PIN_MEMORY,
                shuffle=True,
                drop_last=False,
            )

            scaled_anchors = (
                    torch.tensor(cfg.ANCHORS)
                    * torch.tensor(cfg.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
            )
            scaled_anchors = scaled_anchors.to(cfg.DEVICE)

            utils.plot_examples(model, eval_loader, 0.5, 0.6, scaled_anchors)

            # -- Printing MAP
            pred_boxes, true_boxes = utils.get_evaluation_bboxes(
                eval_loader,
                model,
                iou_threshold=cfg.NMS_IOU_THRESH,
                anchors=cfg.ANCHORS,
                threshold=cfg.CONF_THRESHOLD,
            )
            mapval = utils.mean_average_precision(
                pred_boxes,
                true_boxes,
                iou_threshold=cfg.MAP_IOU_THRESH,
                box_format="midpoint",
                num_classes=cfg.NUM_CLASSES,
            )
            print(f"MAP: {mapval.item()}")
