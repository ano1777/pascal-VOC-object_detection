import gc
import json
import platform
import time
from pathlib import Path
import numpy as np
import torch
from tqdm.auto import tqdm
from .config import seed_everything, write_json, classes_for
from .data import loader
from .metrics import evaluate_detections
from .model import build_model, save_checkpoint, load_checkpoint


@torch.inference_mode()
def predict_loader(model, data_loader, device, amp=False):
    model.eval()
    predictions, targets = [], []
    for images, batch_targets in tqdm(data_loader, desc='Evaluating', leave=False):
        with torch.autocast(device_type=device.type, enabled=amp and device.type == 'cuda'):
            outputs = model([image.to(device) for image in images])
        for out in outputs:
            prediction = {k: v.float().cpu().numpy() if v.is_floating_point()
                          else v.cpu().numpy() for k, v in out.items()}
            if getattr(model, 'task', 'objects') == 'layout':
                from .layout_data import cap_part_predictions
                prediction = cap_part_predictions(prediction)
            predictions.append(prediction)
        targets.extend({k: v.cpu().numpy() for k, v in t.items()} for t in batch_targets)
    return predictions, targets


def train_experiment(cfg, fit_records, tune_records, run_dir, device):
    """Every trial starts from identical pretrained weights and a reset seed."""
    seed_everything(cfg.seed)
    run_dir = Path(run_dir)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f'{run_dir} is not empty; use a new run name.')
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / 'config.json', cfg.dictionary())
    model = build_model(cfg).to(device)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                                lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, cfg.epochs//2), gamma=.3)
    use_amp = cfg.amp and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    train_loader = loader(fit_records, cfg, training=True)
    tune_loader = loader(tune_records, cfg)
    history, best_score, stale = [], -1., 0
    start = time.perf_counter()
    for epoch in range(cfg.epochs):
        model.train()
        totals, batches = {}, 0
        for images, targets in tqdm(train_loader, desc=f'Epoch {epoch+1}/{cfg.epochs}', leave=False):
            images = [image.to(device) for image in images]
            # All valid training boxes, including difficult instances, are supervision.
            targets = [{k: t[k].to(device) for k in ('boxes', 'labels')} for t in targets]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                losses = model(images, targets)
                loss = sum(losses.values())
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss at epoch {epoch+1}: {losses}')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
            scaler.step(optimizer)
            scaler.update()
            batches += 1
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.) + value.item()
        predictions, ground_truth = predict_loader(model, tune_loader, device)
        metrics = evaluate_detections(predictions, ground_truth, thresholds=[.5], classes=classes_for(cfg))
        score = metrics['map50']
        if score is None:
            raise ValueError('Tuning split has no non-difficult objects; increase its size.')
        row = {'epoch': epoch+1, 'lr': optimizer.param_groups[0]['lr'],
               'train_loss': sum(totals.values()) / batches,
               **{k: v/batches for k, v in totals.items()}, 'tune_map50': score}
        history.append(row)
        print(row)
        if score > best_score:
            best_score, stale = score, 0
            save_checkpoint(run_dir / 'best.pt', model, cfg, epoch=epoch+1,
                            tune_map50=score, initialization='COCO_V1; matching class rows copied, new part rows random')
        else:
            stale += 1
        scheduler.step()
        write_json(run_dir / 'history.json', history)
        if stale >= cfg.patience:
            break
    result = {'run': run_dir.name, 'best_tune_map50': best_score,
              'epochs_completed': len(history), 'seconds': time.perf_counter()-start,
              'checkpoint': str((run_dir/'best.pt').resolve()), **cfg.dictionary()}
    write_json(run_dir/'summary.json', result)
    del model, optimizer, scaler
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return result


@torch.inference_mode()
def benchmark(model, dataset, device, warmup=3, repeats=20, amp=False):
    """Batch=1, synchronized model latency; excludes disk/PIL and visualization.

    Input tensors are preloaded, transfer + model resize/NMS are timed. Warm-up
    is excluded. CUDA peak allocation is absolute, including resident weights.
    """
    if not len(dataset) or repeats < 1:
        raise ValueError('Benchmark needs images and at least one repeat.')
    model.eval()
    inputs = [dataset[i][0] for i in range(min(len(dataset), 10))]
    def synchronize():
        if device.type == 'cuda':
            torch.cuda.synchronize()
    def forward(i):
        with torch.autocast(device_type=device.type, enabled=amp and device.type == 'cuda'):
            return model([inputs[i % len(inputs)].to(device)])
    for i in range(warmup):
        forward(i)
    synchronize()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    times = []
    for i in range(repeats):
        synchronize()
        start = time.perf_counter()
        forward(i)
        synchronize()
        times.append(1000 * (time.perf_counter() - start))
    return {'median_ms': float(np.median(times)), 'p95_ms': float(np.percentile(times, 95)),
            'mean_ms': float(np.mean(times)), 'images_per_second': 1000/float(np.mean(times)),
            'repeats': repeats, 'warmup': warmup, 'batch_size': 1,
            'cuda_peak_allocated_mb': torch.cuda.max_memory_allocated()/2**20 if device.type == 'cuda' else None,
            'device': str(device), 'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else platform.processor(),
            'torch_version': str(torch.__version__), 'cpu_threads': torch.get_num_threads(),
            'python_version': platform.python_version(), 'amp': amp and device.type == 'cuda'}
