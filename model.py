from pathlib import Path
import torch
from torch import nn
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops.misc import FrozenBatchNorm2d
from .config import Config, CLASSES, classes_for

ALIASES = {'aeroplane': 'airplane', 'diningtable': 'dining table',
           'motorbike': 'motorcycle', 'pottedplant': 'potted plant',
           'sofa': 'couch', 'tvmonitor': 'tv'}


def freeze_batch_norm(module):
    # A no-download checkpoint reload must have the same normalization topology.
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            frozen = FrozenBatchNorm2d(child.num_features, eps=child.eps)
            with torch.no_grad():
                for key in ('weight', 'bias', 'running_mean', 'running_var'):
                    getattr(frozen, key).copy_(getattr(child, key))
            setattr(module, name, frozen)
        else:
            freeze_batch_norm(child)


def build_model(cfg, pretrained=True):
    weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT if pretrained else None
    arguments = {'weights': weights, 'weights_backbone': None,
                 'min_size': cfg.min_size, 'max_size': cfg.max_size,
                 'box_score_thresh': 0.001, 'box_detections_per_img': 100}
    if pretrained:
        arguments['trainable_backbone_layers'] = 6
    model = fasterrcnn_mobilenet_v3_large_fpn(**arguments)
    freeze_batch_norm(model)
    old_head = model.roi_heads.box_predictor
    head = FastRCNNPredictor(old_head.cls_score.in_features, len(classes_for(cfg)))
    if pretrained:
        # Transfer matching COCO classifier AND box regressor rows. This makes the
        # pretrained baseline meaningful before any VOC training.
        coco_names = weights.meta['categories']
        with torch.no_grad():
            for dst, name in enumerate(classes_for(cfg)):
                coco_name = ALIASES.get(name, name)
                if coco_name not in coco_names:
                    continue  # New body-part rows are randomly initialized and must train.
                src = coco_names.index(coco_name)
                head.cls_score.weight[dst].copy_(old_head.cls_score.weight[src])
                head.cls_score.bias[dst].copy_(old_head.cls_score.bias[src])
                head.bbox_pred.weight[4*dst:4*dst+4].copy_(old_head.bbox_pred.weight[4*src:4*src+4])
                head.bbox_pred.bias[4*dst:4*dst+4].copy_(old_head.bbox_pred.bias[4*src:4*src+4])
    model.roi_heads.box_predictor = head
    for parameter in model.backbone.parameters():
        parameter.requires_grad = not cfg.freeze_backbone
    model.task = cfg.task
    return model


def save_checkpoint(path, model, cfg, **metadata):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({'state_dict': state, 'config': cfg.dictionary(),
                'classes': list(classes_for(cfg)), **metadata}, path)


def load_checkpoint(path, device='cpu'):
    # Only load project-generated, trusted checkpoints. No pretrained download.
    payload = torch.load(path, map_location='cpu', weights_only=True)
    cfg = Config(**payload['config'])
    if payload['classes'] != list(classes_for(cfg)):
        raise ValueError('Checkpoint has a different class mapping.')
    model = build_model(cfg, pretrained=False)
    model.load_state_dict(payload['state_dict'], strict=True)
    return model.to(device).eval(), cfg, payload
