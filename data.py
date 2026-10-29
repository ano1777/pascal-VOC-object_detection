"""Cleaning is deterministic and shared by training, tuning and evaluation."""
import hashlib
import random
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import VOCDetection
from torchvision.transforms.functional import pil_to_tensor
from .config import CLASSES, write_json


def download_voc(root):
    # Torchvision verifies the official archive's MD5 before extraction.
    VOCDetection(str(root), year='2012', image_set='train', download=True)
    return Path(root) / 'VOCdevkit' / 'VOC2012'


def parse_annotation(xml_path, width, height):
    """VOC's 1-based inclusive boxes -> 0-based, half-open XYXY pixels."""
    annotation = ET.parse(xml_path).getroot()
    records, issues = [], []
    size = annotation.find('size')
    if size is not None and (int(size.findtext('width')) != width or
                             int(size.findtext('height')) != height):
        issues.append('xml_size_mismatch_used_decoded_image_size')
    for obj in annotation.findall('object'):
        name = obj.findtext('name')
        if name not in CLASSES[1:]:
            issues.append('unknown_class')
            continue
        box = obj.find('bndbox')
        try:
            values = [float(box.findtext(k)) for k in ('xmin', 'ymin', 'xmax', 'ymax')]
            if not np.isfinite(values).all():
                raise ValueError('nonfinite')
            x1, y1, x2, y2 = values
            clipped = [max(0., min(x1 - 1, width)), max(0., min(y1 - 1, height)),
                       max(0., min(x2, width)), max(0., min(y2, height))]
            if clipped != [x1 - 1, y1 - 1, x2, y2]:
                issues.append('clipped_box')
            if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
                issues.append('degenerate_box')
                continue
            records.append({'box': clipped, 'label': CLASSES.index(name),
                            'difficult': int(obj.findtext('difficult', '0')) != 0})
        except (ValueError, TypeError, AttributeError):
            issues.append('malformed_box')
    return records, issues


def audit_voc(voc_dir, output_dir):
    """Preserve official val first; remove exact decoded-pixel duplicates globally.

    Save all exclusions/repairs rather than silently changing the dataset.
    Images with no valid annotations remain valid background images.
    """
    voc_dir = Path(voc_dir)
    seen, splits, events = {}, {}, []
    for split in ('val', 'train'):
        image_ids = (voc_dir / 'ImageSets/Main' / f'{split}.txt').read_text().split()
        valid = []
        for image_id in image_ids:
            image_path = voc_dir / 'JPEGImages' / f'{image_id}.jpg'
            try:
                with Image.open(image_path) as im:
                    im = im.convert('RGB')
                    width, height = im.size
                    fingerprint = hashlib.sha256(
                        f'{width},{height}:'.encode() + im.tobytes()).hexdigest()
                objects, issues = parse_annotation(
                    voc_dir / 'Annotations' / f'{image_id}.xml', width, height)
                if fingerprint in seen:
                    events.append({'id': image_id, 'split': split,
                                   'issue': 'duplicate_pixels', 'kept': seen[fingerprint]})
                    continue
                seen[fingerprint] = image_id
                valid.append({'id': image_id, 'path': str(image_path.resolve()),
                              'width': width, 'height': height,
                              'sha256_pixels': fingerprint, 'objects': objects})
                events.extend({'id': image_id, 'split': split, 'issue': x} for x in issues)
            except (OSError, ET.ParseError, ValueError, TypeError) as exc:
                events.append({'id': image_id, 'split': split, 'issue': 'excluded_image',
                               'reason': str(exc)})
        splits[split] = valid
    write_json(Path(output_dir) / 'cleaning_report.json', events)
    write_json(Path(output_dir) / 'clean_manifest.json', splits)
    return splits, events


def make_splits(clean, cfg, output_dir, tune_fraction=0.2):
    train = list(clean['train'])
    random.Random(cfg.seed).shuffle(train)
    n_tune = max(1, round(len(train) * tune_fraction))
    fit, tune, evaluation = train[n_tune:], train[:n_tune], list(clean['val'])
    # Limits are applied after splitting; no overlap even in quick mode.
    fit, tune, evaluation = (fit[:cfg.train_limit], tune[:cfg.tune_limit],
                             evaluation[:cfg.eval_limit])
    assert fit and tune and evaluation, 'Each split needs at least one valid image.'
    sets = [set(r['sha256_pixels'] for r in part) for part in (fit, tune, evaluation)]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    parts = {'fit': fit, 'tune': tune, 'evaluation': evaluation}
    write_json(Path(output_dir) / 'splits.json', {
        'seed': cfg.seed, 'config': cfg.dictionary(),
        'image_ids': {k: [r['id'] for r in v] for k, v in parts.items()}})
    return parts


def class_counts(records, classes=CLASSES):
    return {name: sum(o['label'] == label and not o['difficult']
                      for r in records for o in r['objects'])
            for label, name in enumerate(classes) if label}


class VOCDataset(Dataset):
    def __init__(self, records, flip_probability=0.):
        self.records = records
        self.flip_probability = flip_probability

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(record['path']) as im:
            im = im.convert('RGB')
            if 'crop_box' in record:
                im = im.crop(tuple(record['crop_box']))
            image = pil_to_tensor(im).float() / 255.
        objects = record['objects']
        boxes = torch.tensor([o['box'] for o in objects], dtype=torch.float32).reshape(-1, 4)
        if random.random() < self.flip_probability:
            image = image.flip(-1)
            boxes[:, [0, 2]] = image.shape[-1] - boxes[:, [2, 0]]
        target = {'boxes': boxes,
                  'labels': torch.tensor([o['label'] for o in objects], dtype=torch.int64),
                  'difficult': torch.tensor([o['difficult'] for o in objects], dtype=torch.bool),
                  'image_id': torch.tensor(index),
                  'area': (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
                  'iscrowd': torch.zeros(len(objects), dtype=torch.int64)}
        return image, target


def collate(batch):
    return tuple(zip(*batch))


def loader(records, cfg, training=False):
    generator = torch.Generator().manual_seed(cfg.seed)
    return DataLoader(VOCDataset(records, cfg.flip_probability if training else 0),
                      batch_size=cfg.batch_size, shuffle=training,
                      num_workers=cfg.workers, collate_fn=collate, generator=generator,
                      pin_memory=torch.cuda.is_available())
