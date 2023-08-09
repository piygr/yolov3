"""
Implementation of Yolo Loss Function similar to the one in Yolov3 paper,
the difference from what I can tell is I use CrossEntropy for the classes
instead of BinaryCrossEntropy.
"""
import random
import torch
import torch.nn as nn
import pytorch_lightning as pl
from utils import intersection_over_union
import config as cfg


class YoloLoss(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

        # Constants signifying how much to pay for each respective part of the loss
        self.lambda_class = 1
        self.lambda_noobj = 10
        self.lambda_obj = 1
        self.lambda_box = 10

        self.scaled_anchors = (
            torch.tensor(cfg.ANCHORS)
            * torch.tensor(cfg.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
        )

    def forward(self, predictions_list, target_list, **kwargs):

        anchors_list = kwargs.get('anchors_list', None)
        if not anchors_list:
            anchors_list = self.scaled_anchors

        anchors_list = anchors_list.to(cfg.DEVICE)

        box_loss = 0.0
        object_loss = 0.0
        no_object_loss = 0.0
        class_loss = 0.0

        for i in range(3):
            target = target_list[i]
            predictions = predictions_list[i]
            anchors = anchors_list[i]

            # Check where obj and noobj (we ignore if target == -1)
            obj = target[..., 0] == 1  # in paper this is Iobj_i
            noobj = target[..., 0] == 0  # in paper this is Inoobj_i

            # ======================= #
            #   FOR NO OBJECT LOSS    #
            # ======================= #

            no_object_loss += self.bce(
                (predictions[..., 0:1][noobj]), (target[..., 0:1][noobj]),
            )

            # ==================== #
            #   FOR OBJECT LOSS    #
            # ==================== #

            anchors = anchors.reshape(1, 3, 1, 1, 2)
            box_preds = torch.cat([self.sigmoid(predictions[..., 1:3]), torch.exp(predictions[..., 3:5]) * anchors], dim=-1)
            ious = intersection_over_union(box_preds[obj], target[..., 1:5][obj]).detach()
            object_loss += self.mse(self.sigmoid(predictions[..., 0:1][obj]), ious * target[..., 0:1][obj])

            # ======================== #
            #   FOR BOX COORDINATES    #
            # ======================== #

            predictions[..., 1:3] = self.sigmoid(predictions[..., 1:3])  # x,y coordinates
            target[..., 3:5] = torch.log(
                (1e-16 + target[..., 3:5] / anchors)
            )  # width, height coordinates
            box_loss += self.mse(predictions[..., 1:5][obj], target[..., 1:5][obj])

            # ================== #
            #   FOR CLASS LOSS   #
            # ================== #

            class_loss += self.entropy(
                (predictions[..., 5:][obj]), (target[..., 5][obj].long()),
            )

            #print("__________________________________")
            #print(self.lambda_box * box_loss)
            #print(self.lambda_obj * object_loss)
            #print(self.lambda_noobj * no_object_loss)
            #print(self.lambda_class * class_loss)
            #print("\n")

        total_loss = (
            self.lambda_box * box_loss
            + self.lambda_obj * object_loss
            + self.lambda_noobj * no_object_loss
            + self.lambda_class * class_loss
        )

        if kwargs.get('loss_dict'):
            return dict(class_loss=self.lambda_class * class_loss,
                    no_object_loss=self.lambda_noobj * no_object_loss,
                    object_loss=self.lambda_obj * object_loss,
                    box_loss=self.lambda_box * box_loss,
                    total_loss=total_loss
                    )
        else:
            return total_loss


    def check_class_accuracy(self, predictions, target, threshold):
        tot_class_preds, correct_class = 0, 0
        tot_noobj, correct_noobj = 0, 0
        tot_obj, correct_obj = 0, 0

        y = target
        out = predictions

        for i in range(3):
            obj = y[i][..., 0] == 1  # in paper this is Iobj_i
            noobj = y[i][..., 0] == 0  # in paper this is Iobj_i

            correct_class += torch.sum(
                torch.argmax(out[i][..., 5:][obj], dim=-1) == y[i][..., 5][obj]
            )
            tot_class_preds += torch.sum(obj)

            obj_preds = torch.sigmoid(out[i][..., 0]) > threshold
            correct_obj += torch.sum(obj_preds[obj] == y[i][..., 0][obj])
            tot_obj += torch.sum(obj)
            correct_noobj += torch.sum(obj_preds[noobj] == y[i][..., 0][noobj])
            tot_noobj += torch.sum(noobj)

        return dict(
            correct_class=correct_class,
            correct_noobj=correct_noobj,
            correct_obj=correct_obj,
            total_class_preds=tot_class_preds,
            total_noobj=tot_noobj,
            total_obj=tot_obj
        )

        '''print(f"Class accuracy is: {(correct_class/(tot_class_preds+1e-16))*100:2f}%")
        print(f"No obj accuracy is: {(correct_noobj/(tot_noobj+1e-16))*100:2f}%")
        print(f"Obj accuracy is: {(correct_obj/(tot_obj+1e-16))*100:2f}%")'''