from dataset import *
from models.YoloV3Lightning import *
import utils

def init(model, basic_sanity_check=True, find_max_lr=True, **kwargs):
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
            if cfg.LOAD_MODEL:
                trainer = pl.Trainer(
                    checkpoint_path=cfg.CHECKPOINT_FILE
                )
            else:
                trainer = pl.Trainer(
                    precision=16,
                    max_epochs=cfg.NUM_EPOCHS
                )

            train_loader = kwargs.get('train_dataloader')
            test_loader = kwargs.get('test_dataloader')
            trainer.fit(model, train_loader, test_loader)

            scaled_anchors = (
                    torch.tensor(cfg.ANCHORS)
                    * torch.tensor(cfg.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
            )
            utils.plot_couple_examples(model, test_loader, 0.6, 0.5, scaled_anchors)

            pred_boxes, true_boxes = utils.get_evaluation_bboxes(
                test_loader,
                model,
                iou_threshold=config.NMS_IOU_THRESH,
                anchors=config.ANCHORS,
                threshold=config.CONF_THRESHOLD,
            )
            mapval = utils.mean_average_precision(
                pred_boxes,
                true_boxes,
                iou_threshold=config.MAP_IOU_THRESH,
                box_format="midpoint",
                num_classes=config.NUM_CLASSES,
            )
            print(f"MAP: {mapval.item()}")
