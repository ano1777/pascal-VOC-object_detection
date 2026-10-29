"""One inference path shared by Streamlit and the REST API."""
import io
import json
from pathlib import Path
import threading
from PIL import Image, ImageDraw, ImageOps
import torch
from torchvision.transforms.functional import pil_to_tensor
from .config import CLASSES, LAYOUT_CLASSES
from .layout_data import context_crop
from .model import load_checkpoint

MAX_IMAGE_PIXELS = 16_000_000


def decode_image(content):
    with Image.open(io.BytesIO(content)) as opened:
        if opened.width * opened.height > MAX_IMAGE_PIXELS:
            raise ValueError('Use an image smaller than 16 megapixels.')
        return ImageOps.exif_transpose(opened).convert('RGB')


def tensor_image(image, device):
    return (pil_to_tensor(image).float()/255.).to(device)


def select_parts(prediction, offset=(0,0), threshold=.5):
    """One head / up to two hands / up to two feet per supplied person.

    These are cardinality constraints, not proof of body-part ownership.
    Suppressed or missing detections mean 'not detected', not known occlusion.
    """
    limits = {1:1, 2:2, 3:2}
    counts = {1:0, 2:0, 3:0}
    rows = []
    order = torch.argsort(prediction['scores'], descending=True, stable=True)
    for index in order.tolist():
        score = float(prediction['scores'][index])
        label = int(prediction['labels'][index])
        if score < threshold or label not in limits or counts[label] >= limits[label]:
            continue
        box = prediction['boxes'][index].detach().cpu().tolist()
        rows.append({'class': LAYOUT_CLASSES[label], 'confidence': score,
                     'box': [box[0]+offset[0], box[1]+offset[1],
                             box[2]+offset[0], box[3]+offset[1]]})
        counts[label] += 1
    return rows


class DetectorService:
    def __init__(self, artifacts='artifacts', device='cpu'):
        self.artifacts = Path(artifacts)
        self.device = torch.device(device)
        self.models = {}
        self.lock = threading.RLock()

    def readiness(self):
        files = {'objects': ('best.pt','deployment.json'),
                 'layout': ('layout_best.pt','layout_deployment.json')}
        objects = all((self.artifacts/f).exists() for f in files['objects'])
        parts = all((self.artifacts/f).exists() for f in files['layout'])
        return {'objects': objects, 'layout': objects and parts}

    def get_model(self, task):
        if task not in ('objects','layout'):
            raise ValueError('Task must be objects or layout.')
        prefix = '' if task == 'objects' else 'layout_'
        checkpoint = self.artifacts/f'{prefix}best.pt'
        policy_path = self.artifacts/f'{prefix}deployment.json'
        if not checkpoint.exists() or not policy_path.exists():
            raise FileNotFoundError(f'Missing {checkpoint.name} or {policy_path.name}. '
                                    'Run the matching notebook export section first.')
        stamp = (checkpoint.stat().st_mtime_ns, policy_path.stat().st_mtime_ns)
        if task not in self.models or self.models[task][0] != stamp:
            model, cfg, _ = load_checkpoint(checkpoint, self.device)
            if cfg.task != task:
                raise ValueError(f'{checkpoint.name} contains a {cfg.task} model, expected {task}.')
            policy = json.loads(policy_path.read_text())
            model.transform.min_size = (int(policy['min_size']),)
            model.transform.max_size = int(policy['max_size'])
            self.models[task] = (stamp, model, cfg)
        return self.models[task][1:]

    @torch.inference_mode()
    def predict(self, image, task='objects', threshold=.5, person_threshold=.5, max_people=20):
        if task not in ('objects','layout'):
            raise ValueError('Task must be objects or layout.')
        if not 0 <= threshold <= 1 or not 0 <= person_threshold <= 1:
            raise ValueError('Threshold must be between zero and one.')
        if max_people < 1:
            raise ValueError('max_people must be positive.')
        image = ImageOps.exif_transpose(image).convert('RGB')
        if image.width * image.height > MAX_IMAGE_PIXELS:
            raise ValueError('Use an image smaller than 16 megapixels.')
        # Serial execution also prevents cache reload while another request uses it.
        with self.lock:
            object_model, _ = self.get_model('objects')
            parts_model, layout_cfg = self.get_model('layout') if task == 'layout' else (None,None)
            prediction = object_model([tensor_image(image, self.device)])[0]
            objects = []
            for box, score, label in zip(prediction['boxes'], prediction['scores'], prediction['labels']):
                if float(score) >= (person_threshold if task == 'layout' else threshold):
                    objects.append({'class': CLASSES[int(label)], 'confidence': float(score),
                                    'box': box.detach().cpu().tolist()})
            base = {'task': task, 'width': image.width, 'height': image.height,
                    'coordinates': 'zero-based XYXY pixels, half-open'}
            if task == 'objects':
                return {**base, 'detections': objects}
            people = sorted((o for o in objects if o['class']=='person'),
                            key=lambda p:-p['confidence'])
            layouts = []
            for index, person in enumerate(people[:max_people], start=1):
                crop = context_crop(person['box'], image.width, image.height, layout_cfg.crop_padding)
                if crop[2] <= crop[0] or crop[3] <= crop[1]:
                    continue
                cropped = image.crop(tuple(crop))
                part_prediction = parts_model([tensor_image(cropped, self.device)])[0]
                layouts.append({'person_id': index, 'box': person['box'],
                                'confidence': person['confidence'],
                                'parts': select_parts(part_prediction, crop[:2], threshold)})
            return {**base, 'people': layouts, 'people_above_threshold': len(people),
                    'people_limit': max_people, 'truncated': len(people)>max_people}


def draw_result(image, result):
    image = image.convert('RGB').copy()
    drawing = ImageDraw.Draw(image)
    colors = {'person':'#00bfa5', 'head':'#ffd740', 'hand':'#40c4ff', 'foot':'#ff80ab'}
    def draw(box, label, color):
        drawing.rectangle(box, outline=color, width=3)
        drawing.text((box[0]+3, max(0,box[1]+3)), label,
                     fill='white', stroke_width=1, stroke_fill='black')
    if result['task'] == 'objects':
        for obj in result['detections']:
            draw(obj['box'], f"{obj['class']} {obj['confidence']:.2f}", '#00bfa5')
    else:
        for person in result['people']:
            draw(person['box'], f"Person {person['person_id']}", colors['person'])
            for part in person['parts']:
                draw(part['box'], f"P{person['person_id']} {part['class']} {part['confidence']:.2f}",
                     colors[part['class']])
    return image
