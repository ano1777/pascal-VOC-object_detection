"""VOC continuous interpolated AP, with difficult instances ignored.

This is NOT the COCO API: AP50:95 here averages VOC-style AP over ten IoUs.
Half-open coordinates avoid the inclusive-pixel '+1' convention internally.
"""
import numpy as np
from .config import CLASSES


def iou_one(box, boxes):
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    if not len(boxes):
        return np.empty(0)
    intersection = np.maximum(0, np.minimum(box[2:], boxes[:, 2:]) -
                              np.maximum(box[:2], boxes[:, :2])).prod(axis=1)
    area_a = np.maximum(0, np.asarray(box)[2:] - np.asarray(box)[:2]).prod()
    area_b = np.maximum(0, boxes[:, 2:] - boxes[:, :2]).prod(axis=1)
    return intersection / np.maximum(area_a + area_b - intersection, 1e-12)


def interpolated_ap(recall, precision):
    r = np.r_[0., recall, 1.]
    p = np.r_[0., precision, 0.]
    p = np.maximum.accumulate(p[::-1])[::-1]
    changes = np.where(r[1:] != r[:-1])[0]
    return float(np.sum((r[changes+1] - r[changes]) * p[changes+1]))


def evaluate_detections(predictions, targets, thresholds=None, classes=CLASSES):
    if len(predictions) != len(targets):
        raise ValueError('Predictions and targets must have equal length.')
    thresholds = np.round(np.arange(.5, .96, .05), 2) if thresholds is None else thresholds
    by_iou, curves = {}, {}
    for threshold in thresholds:
        class_ap = {}
        for label, name in enumerate(classes[1:], start=1):
            ground_truth, positives = {}, 0
            detections = []
            for image_id, (prediction, target) in enumerate(zip(predictions, targets)):
                mask = np.asarray(target['labels']) == label
                difficult = np.asarray(target['difficult'], dtype=bool)[mask]
                ground_truth[image_id] = {
                    'boxes': np.asarray(target['boxes']).reshape(-1, 4)[mask],
                    'difficult': difficult, 'used': np.zeros(mask.sum(), dtype=bool)}
                positives += int((~difficult).sum())
                for box, score, category in zip(prediction['boxes'], prediction['scores'], prediction['labels']):
                    if int(category) == label:
                        detections.append((float(score), image_id, np.asarray(box)))
            detections.sort(key=lambda d: -d[0])  # Stable ties.
            tp, fp = [], []
            for score, image_id, box in detections:
                gt = ground_truth[image_id]
                overlaps = iou_one(box, gt['boxes'])
                best = int(overlaps.argmax()) if len(overlaps) else -1
                if best >= 0 and overlaps[best] > threshold:
                    if gt['difficult'][best]:
                        continue  # Neither TP nor FP, even for repeated detections.
                    if not gt['used'][best]:
                        gt['used'][best] = True
                        tp.append(1); fp.append(0)
                    else:
                        tp.append(0); fp.append(1)
                else:
                    tp.append(0); fp.append(1)
            cumulative_tp, cumulative_fp = np.cumsum(tp), np.cumsum(fp)
            recall = cumulative_tp / max(positives, 1)
            precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1)
            class_ap[name] = interpolated_ap(recall, precision) if positives else None
            if abs(threshold - .5) < 1e-6:
                curves[name] = {'recall': recall.tolist(), 'precision': precision.tolist(),
                                'positives': positives}
        by_iou[f'{threshold:.2f}'] = class_ap
    means = {key: float(np.mean([v for v in values.values() if v is not None]))
             if any(v is not None for v in values.values()) else None
             for key, values in by_iou.items()}
    valid = [v for v in means.values() if v is not None]
    return {'map50': means.get('0.50'),
            'map50_95_voc': float(np.mean(valid)) if valid else None,
            'per_class_ap50': by_iou.get('0.50', {}), 'map_by_iou': means,
            'classes_with_gt': sum(v is not None for v in next(iter(by_iou.values())).values()),
            'pr_curves': curves, 'images': len(targets)}


def error_analysis(predictions, targets, score_threshold=.5, match_iou=.5):
    """Greedy diagnostic categories at one operating point, not an AP metric."""
    rows = []
    for image_id, (pred, gt) in enumerate(zip(predictions, targets)):
        boxes = np.asarray(gt['boxes']).reshape(-1, 4)
        labels = np.asarray(gt['labels'])
        difficult = np.asarray(gt['difficult'], dtype=bool)
        used = np.zeros(len(boxes), dtype=bool)
        counts = dict(tp=0, duplicate=0, wrong_class=0, localization=0,
                      background=0, ignored_difficult=0, missed=0)
        order = np.argsort(-np.asarray(pred['scores']), kind='stable')
        for j in order:
            if pred['scores'][j] < score_threshold:
                continue
            overlaps = iou_one(np.asarray(pred['boxes'][j]), boxes)
            same = np.where(labels == pred['labels'][j])[0]
            best_same = same[np.argmax(overlaps[same])] if len(same) else -1
            if best_same >= 0 and overlaps[best_same] > match_iou:
                if difficult[best_same]:
                    counts['ignored_difficult'] += 1
                elif used[best_same]:
                    counts['duplicate'] += 1
                else:
                    used[best_same] = True
                    counts['tp'] += 1
            elif len(overlaps) and overlaps.max() > match_iou:
                counts['wrong_class'] += 1
            elif len(same) and overlaps[same].max() > .1:
                counts['localization'] += 1
            else:
                counts['background'] += 1
        counts['missed'] = int((~used & ~difficult).sum())
        rows.append({'image_index': image_id, **counts})
    return rows
