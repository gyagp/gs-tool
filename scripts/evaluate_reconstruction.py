"""Evaluate Brush's held-out renders on foreground pixels; save local comparisons.

Run with the isolated reconstruction Python. No network calls. Full-image scores
are reported separately because empty backgrounds can inflate those scores.
"""
import argparse
import json
import shutil
from pathlib import Path
import cv2
import numpy as np


def metrics(reference, prediction, foreground):
    a, b = reference.astype(np.float32) / 255, prediction.astype(np.float32) / 255
    error = np.square(a - b).mean(axis=2)
    def psnr(mask):
        return float(-10 * np.log10(max(1e-12, error[mask].mean())))
    blur = lambda im: cv2.GaussianBlur(im, (11, 11), 1.5)
    ma, mb = blur(a), blur(b)
    va, vb, cov = blur(a * a) - ma * ma, blur(b * b) - mb * mb, blur(a * b) - ma * mb
    ssim = ((2 * ma * mb + .01 ** 2) * (2 * cov + .03 ** 2) /
            ((ma * ma + mb * mb + .01 ** 2) * (va + vb + .03 ** 2))).mean(axis=2)
    return {'foreground_psnr': psnr(foreground), 'foreground_ssim': float(ssim[foreground].mean()),
            'full_image_psnr': psnr(np.ones_like(foreground))}


def evaluate_images(root, folder):
    views = []
    for path in sorted(folder.glob('*.png')):
        truth = cv2.imread(str(root / 'dataset/images' / path.name), cv2.IMREAD_UNCHANGED)
        predicted = cv2.imread(str(path))
        if truth is None or predicted is None or truth.shape[:2] != predicted.shape[:2]:
            raise ValueError(f'Missing reference or mismatched render: {path}')
        alpha = truth[:, :, 3].astype(float) / 255
        reference = np.rint(truth[:, :, :3] * alpha[:, :, None]).astype(np.uint8)
        foreground = cv2.erode((alpha > .95).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        if foreground.sum() < 32:
            raise ValueError(f'No evaluable foreground: {path}')
        views.append({'name': path.name, **metrics(reference, predicted, foreground)})
    if not views:
        raise ValueError(f'No renders in {folder}')
    return {'views': views, 'mean': {k: float(np.mean([v[k] for v in views])) for k in views[0] if k != 'name'}}


def evaluate(root, training):
    folders = sorted((root / training).glob('eval_*'), key=lambda p: int(p.name.split('_')[1]))
    reports = []
    if not folders:
        raise ValueError('No saved held-out renders. Train with --eval-save-to-disk.')
    for folder in folders:
        reports.append({'step': int(folder.name.split('_')[1]), **evaluate_images(root, folder)})
    out = root / training / 'validation'
    out.mkdir(exist_ok=True)
    (out / 'report.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')
    best = max(reports, key=lambda report: report['mean']['foreground_ssim'])
    checkpoint = next((path for path in (root / training).glob('export_*.ply')
                       if path.stem.removeprefix('export_').isdigit() and int(path.stem.removeprefix('export_')) == best['step']), None)
    if checkpoint:
        shutil.copyfile(checkpoint, out / 'best.ply')
        (out / 'best.json').write_text(json.dumps({'checkpoint': checkpoint.name, **best}, indent=2), encoding='utf-8')
    # Up to six held-out views across the orbit, source versus early and final.
    names = [p.name for p in sorted(folders[-1].glob('*.png'))]
    names = [names[i] for i in np.linspace(0, len(names) - 1, min(6, len(names))).astype(int)]
    selected = list(dict.fromkeys([folders[0], folders[-1]]))
    panels = []
    for name in names:
        truth = cv2.imread(str(root / 'dataset/images' / name), cv2.IMREAD_UNCHANGED)
        source = (truth[:, :, :3].astype(float) * (truth[:, :, 3:4] / 255)).astype(np.uint8)
        mask = truth[:, :, 3] > 127
        ys, xs = np.where(mask)
        bounds = (max(0, xs.min() - 10), max(0, ys.min() - 10), min(truth.shape[1], xs.max() + 11), min(truth.shape[0], ys.max() + 11))
        row = []
        for label, im in [('Source (held out)', source)] + [(f.name, cv2.imread(str(f / name))) for f in selected]:
            x0, y0, x1, y1 = bounds
            crop = im[y0:y1, x0:x1]
            scale = min(360 / crop.shape[0], 260 / crop.shape[1])
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            tile = np.full((408, 280, 3), 24, np.uint8)
            x, y = (280 - crop.shape[1]) // 2, 45
            tile[y:y + crop.shape[0], x:x + crop.shape[1]] = crop
            cv2.putText(tile, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (235, 235, 235), 1, cv2.LINE_AA)
            cv2.putText(tile, name, (8, 39), cv2.FONT_HERSHEY_SIMPLEX, .4, (160, 160, 160), 1, cv2.LINE_AA)
            row.append(tile)
        panels.append(np.hstack(row))
    cv2.imwrite(str(out / 'held-out-comparison.jpg'), np.vstack(panels))
    print(json.dumps([{k: v for k, v in r.items() if k != 'views'} for r in reports], indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--training-name', default='trained')
    parser.add_argument('--renders', type=Path, help='Evaluate a directory of calibrated web renders instead of training checkpoints')
    args = parser.parse_args()
    if args.renders:
        report = evaluate_images(args.output, args.renders)
        (args.renders / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report['mean'], indent=2))
    else:
        evaluate(args.output, args.training_name)
