"""Read ONLY persons explicitly listed in ImageSets/Layout, not Main.

Person-indexed crops retain part ownership and avoid treating every unannotated
person in a full image as an annotated negative. Neighboring people may still
appear in crops; this is a limitation, especially in crowded scenes.
"""
import hashlib
import math
import random
from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
from .config import LAYOUT_CLASSES, write_json


def context_crop(person_box, width, height, padding=.35):
    if padding < 0:
        raise ValueError('Crop padding must be nonnegative.')
    x1, y1, x2, y2 = person_box
    dx, dy = (x2-x1)*padding, (y2-y1)*padding
    return [max(0, math.floor(x1-dx)), max(0, math.floor(y1-dy)),
            min(width, math.ceil(x2+dx)), min(height, math.ceil(y2+dy))]


def read_box(element, width, height):
    box = element.find('bndbox')
    if box is None:
        raise ValueError('Missing bounding box.')
    raw = [float(box.findtext(k)) for k in ('xmin', 'ymin', 'xmax', 'ymax')]
    if not np.isfinite(raw).all():
        raise ValueError('Nonfinite coordinates.')
    converted = [raw[0]-1, raw[1]-1, raw[2], raw[3]]
    clipped = [max(0., min(converted[0], width)), max(0., min(converted[1], height)),
               max(0., min(converted[2], width)), max(0., min(converted[3], height))]
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        raise ValueError('Degenerate bounding box.')
    return clipped, clipped != converted


def read_person(xml_path, object_index, width, height, padding=.35):
    """The 1-based object index addresses ALL top-level objects, not just people."""
    objects = ET.parse(xml_path).getroot().findall('object')
    if not 1 <= object_index <= len(objects):
        raise ValueError('Object index outside XML annotation.')
    person = objects[object_index-1]
    if person.findtext('name') != 'person':
        raise ValueError('Layout index does not refer to a person.')
    person_box, repaired = read_box(person, width, height)
    issues = ['clipped_person_box'] if repaired else []
    crop = context_crop(person_box, width, height, padding)
    parts = []
    for part in person.findall('part'):
        name = part.findtext('name')
        if name not in LAYOUT_CLASSES[1:]:
            # Reject the whole sample instead of silently turning malformed labels
            # into background supervision.
            raise ValueError(f'Unknown part class: {name}')
        box, repaired = read_box(part, width, height)
        if repaired:
            issues.append('clipped_part_to_image')
        if box[0] < crop[0] or box[1] < crop[1] or box[2] > crop[2] or box[3] > crop[3]:
            raise ValueError('Part outside fixed context crop; increase crop_padding and rerun audit.')
        relative = [box[0]-crop[0], box[1]-crop[1], box[2]-crop[0], box[3]-crop[1]]
        parts.append({'box': relative, 'label': LAYOUT_CLASSES.index(name), 'difficult': False})
    counts = Counter(p['label'] for p in parts)
    if counts[1] != 1 or counts[2] > 2 or counts[3] > 2 or len(parts) < 2:
        raise ValueError('Expected one head and at least one hand/foot, at most two hands/feet.')
    return {'person_box': person_box, 'crop_box': crop, 'objects': parts}, issues


def audit_layout(voc_dir, output_dir, padding=.35):
    """Deduplicate images globally, retaining every listed person in kept images."""
    voc_dir, output_dir = Path(voc_dir), Path(output_dir)
    seen_pixels, splits, events = {}, {}, []
    for split in ('val', 'train'):
        split_path = voc_dir/'ImageSets/Layout'/f'{split}.txt'
        if not split_path.exists():
            raise FileNotFoundError(f'{split_path}: use the full VOC2012 train/val archive.')
        grouped = {}
        for line in split_path.read_text().splitlines():
            if not line.strip():
                continue
            image_id, object_index = line.split()
            grouped.setdefault(image_id, set()).add(int(object_index))
        records = []
        for image_id, indices in grouped.items():
            path = voc_dir/'JPEGImages'/f'{image_id}.jpg'
            try:
                with Image.open(path) as image:
                    image = image.convert('RGB')
                    width, height = image.size
                    digest = hashlib.sha256(f'{width},{height}:'.encode()+image.tobytes()).hexdigest()
                if digest in seen_pixels:
                    events.append({'id': image_id, 'split': split, 'issue': 'duplicate_image',
                                   'kept': seen_pixels[digest]})
                    continue
                seen_pixels[digest] = image_id
            except OSError as exc:
                events.append({'id': image_id, 'split': split, 'issue': 'excluded_image', 'reason': str(exc)})
                continue
            for index in sorted(indices):
                try:
                    person, issues = read_person(voc_dir/'Annotations'/f'{image_id}.xml',
                                                  index, width, height, padding)
                    records.append({'id': f'{image_id}:{index}', 'source_image_id': image_id,
                                    'object_index': index, 'path': str(path.resolve()),
                                    'width': width, 'height': height, 'sha256_pixels': digest,
                                    **person})
                    events.extend({'id': f'{image_id}:{index}', 'split': split, 'issue': issue}
                                  for issue in issues)
                except (OSError, ET.ParseError, ValueError, TypeError) as exc:
                    events.append({'id': f'{image_id}:{index}', 'split': split,
                                   'issue': 'excluded_person', 'reason': str(exc)})
        splits[split] = records
    write_json(output_dir/'layout_cleaning_report.json', events)
    write_json(output_dir/'layout_manifest.json', splits)
    return splits, events


def make_layout_splits(clean, cfg, output_dir, tune_fraction=.2):
    groups = {}
    for record in clean['train']:
        groups.setdefault(record['sha256_pixels'], []).append(record)
    keys = sorted(groups)
    random.Random(cfg.seed).shuffle(keys)
    if len(keys) < 2:
        raise ValueError('Need at least two distinct training images for fit/tune splitting.')
    tune_keys = set(keys[:max(1, min(len(keys)-1, round(len(keys)*tune_fraction)))])
    parts = {
        'fit': [r for k in keys if k not in tune_keys for r in groups[k]],
        'tune': [r for k in keys if k in tune_keys for r in groups[k]],
        'evaluation': list(clean['val'])}
    for name, limit in zip(parts, (cfg.train_limit, cfg.tune_limit, cfg.eval_limit)):
        parts[name] = parts[name][:limit]
    if not all(parts.values()):
        raise ValueError('All splits must contain a person. Inspect exclusions and limits.')
    fingerprints = [set(r['sha256_pixels'] for r in part) for part in parts.values()]
    if any(fingerprints[i] & fingerprints[j] for i,j in ((0,1),(0,2),(1,2))):
        raise ValueError('Image overlap across layout splits.')
    write_json(Path(output_dir)/'layout_splits.json', {
        'config': cfg.dictionary(), 'person_ids': {k:[r['id'] for r in v] for k,v in parts.items()},
        'unique_images': {k:len(set(r['source_image_id'] for r in v)) for k,v in parts.items()}})
    return parts


def cap_part_predictions(prediction):
    """Same per-person cardinalities as serving, retaining score-floor outputs for AP."""
    limits, counts, keep = {1:1, 2:2, 3:2}, {1:0, 2:0, 3:0}, []
    for i in np.argsort(-np.asarray(prediction['scores']), kind='stable'):
        label = int(prediction['labels'][i])
        if label in limits and counts[label] < limits[label]:
            keep.append(int(i))
            counts[label] += 1
    return {key: np.asarray(value)[keep] for key, value in prediction.items()}
