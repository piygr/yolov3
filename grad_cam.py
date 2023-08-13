from typing import List
import torch
import numpy as np
import utils
from pytorch_grad_cam.base_cam import BaseCAM
from pytorch_grad_cam.utils import get_2d_projection
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

class YoloGradCAM(BaseCAM):
    def __init__(self,
                 model,
                 target_layers,
                 scaled_anchors,
                 use_cuda=False,
                 reshape_transform=None):
        super(YoloGradCAM, self).__init__(model,
                                      target_layers,
                                      use_cuda,
                                      reshape_transform,
                                      uses_gradients=False)

        self.scaled_anchors = scaled_anchors

    def get_cam_image(self,
                      input_tensor: torch.Tensor,
                      target_layer: torch.nn.Module,
                      targets: List[torch.nn.Module],
                      activations: torch.Tensor,
                      grads: torch.Tensor,
                      eigen_smooth: bool = False) -> np.ndarray:
        return get_2d_projection(activations)

    def forward(self,
                input_tensor: torch.Tensor,
                targets: List[torch.nn.Module],
                eigen_smooth: bool = False) -> np.ndarray:

        if self.cuda:
            input_tensor = input_tensor.cuda()

        if self.compute_input_gradient:
            input_tensor = torch.autograd.Variable(input_tensor,
                                                   requires_grad=True)

        outputs = self.activations_and_grads(input_tensor)
        if targets is None:
            bboxes = [[] for _ in range(1)]
            for i in range(3):
                batch_size, A, S, _, _ = outputs[i].shape
                anchor = self.scaled_anchors[i]
                boxes_scale_i = utils.cells_to_bboxes(
                    outputs[i], anchor, S=S, is_preds=True
                )
                for idx, (box) in enumerate(boxes_scale_i):
                    bboxes[idx] += box

            nms_boxes = utils.non_max_suppression(
                bboxes[0], iou_threshold=0.5, threshold=0.4, box_format="midpoint",
            )
            # target_categories = np.argmax(outputs.cpu().data.numpy(), axis=-1)
            target_categories = [box[0] for box in nms_boxes]
            targets = [ClassifierOutputTarget(
                category) for category in target_categories]

        if self.uses_gradients:
            self.model.zero_grad()
            loss = sum([target(output)
                        for target, output in zip(targets, outputs)])
            loss.backward(retain_graph=True)

        # In most of the saliency attribution papers, the saliency is
        # computed with a single target layer.
        # Commonly it is the last convolutional layer.
        # Here we support passing a list with multiple target layers.
        # It will compute the saliency image for every image,
        # and then aggregate them (with a default mean aggregation).
        # This gives you more flexibility in case you just want to
        # use all conv layers for example, all Batchnorm layers,
        # or something else.
        cam_per_layer = self.compute_cam_per_layer(input_tensor,
                                                   targets,
                                                   eigen_smooth)
        return self.aggregate_multi_layers(cam_per_layer)