from PIL import Image, ImageDraw, ImageOps
from .config import CLASSES


def draw_detections(image, prediction, threshold=.5, classes=CLASSES):
    image = ImageOps.exif_transpose(image).convert('RGB').copy()
    drawing = ImageDraw.Draw(image)
    for box, score, label in zip(prediction['boxes'], prediction['scores'], prediction['labels']):
        if float(score) < threshold:
            continue
        drawing.rectangle([float(v) for v in box], outline='#00d49b', width=3)
        drawing.text((float(box[0])+3, float(box[1])+3),
                     f'{classes[int(label)]} {float(score):.2f}', fill='white', stroke_width=1, stroke_fill='black')
    return image
