"""Local measured-pose reconstruction. Media never leaves this computer.

Unlike the browser's round-bottle approximation, this pipeline estimates intrinsics,
camera poses and sparse geometry with COLMAP before training actual 3D Gaussians.
Run --help. Use --stages to resume; failed pose estimation never invents an orbit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def progress(root, stage, **details):
    value = {"stage": stage, "updated": time.time(), **details}
    write_json(root / "progress.json", value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def extract(args, root):
    manifest_path = root / "frames.json"
    digest = sha256(args.video)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest["source_sha256"] != digest or manifest["target_frames"] != args.frames
                or manifest.get("max_dimension", 1920) != args.max_dimension):
            raise ValueError("Output belongs to a different video/configuration; choose a fresh --output directory")
        if not all((root / 'frames' / frame['name']).is_file() for frame in manifest['frames']):
            raise ValueError('Extracted frames are missing; choose a fresh output directory')
        return manifest
    cap = cv2.VideoCapture(str(args.video))
    fps, count = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if not cap.isOpened() or fps <= 0 or count < 2:
        raise ValueError("Could not decode the video")
    duration = count / fps
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    images = root / "frames"
    images.mkdir(exist_ok=True)
    frames, previous = [], None
    try:
        for section in range(args.frames):
            candidates = []
            for candidate in range(3):
                seconds = (section + (candidate + .5) / 3) * duration / args.frames
                cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
                ok, image = cap.read()
                if not ok:
                    continue
                scale = min(1., args.max_dimension / max(image.shape[:2])) if args.max_dimension else 1.
                if scale < 1:
                    image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                h, w = gray.shape
                subject = gray[h // 10:9 * h // 10, w // 4:3 * w // 4]
                focus = float(cv2.Laplacian(subject, cv2.CV_32F).var())
                clipped = float(((subject < 8) | (subject > 250)).mean())
                score = math.log1p(focus) - clipped * 2
                candidates.append((score, seconds, image, cv2.resize(gray, (48, 48))))
            if not candidates:
                continue
            score, seconds, image, fingerprint = max(candidates, key=lambda x: x[0])
            # Keep fine-baseline motion for SfM, rejecting truly stationary footage.
            if previous is not None and np.abs(fingerprint.astype(float) - previous).mean() < .4:
                continue
            name = f"frame_{section:05d}.png"
            if not cv2.imwrite(str(images / name), image):
                raise OSError(f"Failed writing {name}")
            frames.append({"name": name, "time": seconds, "focus_score": score})
            previous = fingerprint.astype(float)
            progress(root, "extract", completed=section + 1, total=args.frames, retained=len(frames))
    finally:
        cap.release()
    if len(frames) < 16:
        raise ValueError("Too few distinct video frames for measured-pose reconstruction")
    manifest = {"source_sha256": digest, "source_resolution": [width, height], "fps": fps,
                "duration": duration, "target_frames": args.frames, "max_dimension": args.max_dimension, "frames": frames}
    write_json(manifest_path, manifest)
    return manifest


def sfm(args, root):
    import pycolmap
    sparse = root / "sfm"
    sparse.mkdir(exist_ok=True)
    database = root / "colmap.db"
    extraction = pycolmap.FeatureExtractionOptions()
    extraction.num_threads = args.threads
    extraction.max_image_size = args.max_dimension or 4096
    extraction.sift.max_num_features = 8192
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "SIMPLE_RADIAL"
    progress(root, "features")
    pycolmap.extract_features(database, root / "frames", camera_mode=pycolmap.CameraMode.SINGLE,
                             reader_options=reader, extraction_options=extraction, device=pycolmap.Device.cpu)
    matching = pycolmap.FeatureMatchingOptions()
    matching.num_threads = args.threads
    pairing = pycolmap.SequentialPairingOptions()
    pairing.overlap = 12
    pairing.quadratic_overlap = True
    progress(root, "matching")
    pycolmap.match_sequential(database, matching_options=matching, pairing_options=pairing, device=pycolmap.Device.cpu)
    # Short orbit sequences need loop closure; sequential-only matching split the
    # reference orbit into disconnected models. Avoid silently losing half a side.
    if args.exhaustive or args.frames <= 160:
        progress(root, "matching-all-pairs")
        pycolmap.match_exhaustive(database, matching_options=matching, device=pycolmap.Device.cpu)
    options = pycolmap.IncrementalPipelineOptions()
    options.num_threads = args.threads
    options.random_seed = 42
    options.ba_use_gpu = False
    options.mapper.init_min_tri_angle = 8
    options.mapper.filter_max_reproj_error = 3
    # Keep intermediate models inspectable during long CPU bundle adjustment.
    options.snapshot_path = str(root / "snapshots")
    options.snapshot_frames_freq = 30
    (root / "snapshots").mkdir(exist_ok=True)
    progress(root, "poses")
    models = pycolmap.incremental_mapping(database, root / "frames", sparse, options=options,
        next_image_callback=lambda: progress(root, "poses", registered_update=time.time()))
    if not models:
        raise ValueError("COLMAP could not estimate camera poses; no synthetic orbit was substituted")
    model = max(models.values(), key=lambda m: m.num_reg_images())
    total = len(json.loads((root / "frames.json").read_text(encoding="utf-8"))["frames"])
    report = {"registered": model.num_reg_images(), "frames": total,
              "points": model.num_points3D(), "reprojection_error": model.compute_mean_reprojection_error(),
              "cameras": [c.summary() for c in model.cameras.values()]}
    write_json(root / "sfm-report.json", report)
    if model.num_reg_images() < max(16, total * .9):
        raise ValueError(f"Insufficient measured poses: {report}; inspect video/feature matches")
    (root / "measured").mkdir(exist_ok=True)
    model.write(str(root / "measured"))
    progress(root, "poses-ready", **report)


def segment(args, root):
    import onnxruntime as ort
    images = sorted((root / "frames").glob("*.png"))
    masks = root / "masks"
    masks.mkdir(exist_ok=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    session = ort.InferenceSession(str(args.segmentation_model), options, providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    reports = []
    for index, path in enumerate(images):
        target = masks / path.name
        if target.exists():
            continue
        image = cv2.imread(str(path))
        rgb = cv2.cvtColor(cv2.resize(image, (320, 320)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        tensor = ((rgb - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)).transpose(2, 0, 1)[None]
        prediction = session.run(None, {name: tensor})[0].squeeze()
        prediction = (prediction - prediction.min()) / max(1e-6, float(np.ptp(prediction)))
        soft = cv2.resize(prediction, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
        # Keep the dominant central connected component, never an invented silhouette.
        n, labels, stats, centers = cv2.connectedComponentsWithStats((soft > .5).astype(np.uint8))
        if n < 2:
            raise ValueError(f"No foreground found in {path.name}")
        h, w = soft.shape
        scores = [stats[i, cv2.CC_STAT_AREA] / (1 + 4 * np.linalg.norm((centers[i] - [w / 2, h / 2]) / [w, h])) for i in range(1, n)]
        selected = 1 + int(np.argmax(scores))
        component = (labels == selected).astype(np.uint8)
        # Retain soft boundary pixels only close to the observed foreground.
        region = cv2.dilate(component, np.ones((5, 5), np.uint8))
        alpha = (np.clip((soft - .2) / .6, 0, 1) * region * 255).astype(np.uint8)
        area = float((alpha > 127).mean())
        if not .003 < area < .85:
            raise ValueError(f"Unusable foreground in {path.name}: area={area}")
        cv2.imwrite(str(target), alpha)
        reports.append({"name": path.name, "area": area, "bounds": stats[selected, :4].tolist()})
        progress(root, "segment", completed=index + 1, total=len(images))
    if reports:
        write_json(root / "mask-report.json", reports)


def prepare(args, root):
    import pycolmap
    dataset = root / "dataset"
    dataset.mkdir(exist_ok=True)
    # Brush expects pinhole images. Undistort BOTH colors and masks with identical
    # intrinsics, rather than silently dropping the estimated radial distortion.
    progress(root, "undistort")
    pycolmap.undistort_images(dataset, root / "measured", root / "frames", num_threads=args.threads)
    source = pycolmap.Reconstruction(root / "measured")
    model = pycolmap.Reconstruction(dataset / "sparse")
    for index, image in enumerate(model.images.values()):
        camera = model.cameras[image.camera_id]
        old_image = source.images[image.image_id]
        old_camera = source.cameras[old_image.camera_id]
        old_mask = cv2.imread(str(root / "masks" / image.name), cv2.IMREAD_GRAYSCALE)
        if old_mask is None:
            raise ValueError(f"Missing foreground: {image.name}")
        k = old_camera.calibration_matrix()
        params = old_camera.params
        distortion = np.array([params[3], 0, 0, 0, 0])
        mapx, mapy = cv2.initUndistortRectifyMap(k, distortion, None, camera.calibration_matrix(),
                                               (camera.width, camera.height), cv2.CV_32FC1)
        mask = cv2.remap(old_mask, mapx, mapy, cv2.INTER_LINEAR)
        path = dataset / "images" / image.name
        color = cv2.imread(str(path))
        if color is None or color.shape[:2] != mask.shape:
            raise ValueError(f"Undistorted color/mask dimensions disagree: {image.name}")
        cv2.imwrite(str(path), np.dstack((color, mask)))
        progress(root, "prepare-alpha", completed=index + 1, total=model.num_reg_images())

    ids = list(model.points3D)
    xyz = np.array([model.points3D[i].xyz for i in ids])
    visible = np.zeros(len(ids), int)
    inside = np.zeros(len(ids), int)
    for image in model.images.values():
        camera = model.cameras[image.camera_id]
        pose = image.cam_from_world().matrix()
        cam_xyz = xyz @ pose[:, :3].T + pose[:, 3]
        uv = camera.img_from_cam(cam_xyz)
        valid = (cam_xyz[:, 2] > 0) & np.isfinite(uv).all(axis=1)
        valid &= (uv[:, 0] >= 0) & (uv[:, 0] < camera.width - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < camera.height - 1)
        points = np.where(valid)[0]
        xy = np.rint(uv[points]).astype(int)
        alpha = cv2.imread(str(dataset / "images" / image.name), cv2.IMREAD_UNCHANGED)[:, :, 3]
        visible[points] += 1
        inside[points] += alpha[xy[:, 1], xy[:, 0]] > 96
    keep = (visible >= 8) & (inside / np.maximum(visible, 1) >= .8)
    if keep.sum() < 32:
        raise ValueError(f"Only {keep.sum()} sparse points agree with foreground masks; inspect segmentation and poses")
    for i, good in zip(ids, keep):
        if not good:
            model.delete_point3D(i)

    center = np.median(xyz[keep], axis=0)
    first = min(model.images.values(), key=lambda im: im.name)
    rotations = [im.cam_from_world().rotation.matrix().T for im in model.images.values()]
    down = np.median([rotation[:, 1] for rotation in rotations], axis=0)
    down /= np.linalg.norm(down)
    right = first.cam_from_world().rotation.matrix().T[:, 0].copy()
    right -= np.dot(right, down) * down
    right /= np.linalg.norm(right)
    forward = np.cross(right, down)
    rotation = np.array([right, down, forward])
    upright = (xyz[keep] - center) @ rotation.T
    extent = np.max(np.quantile(upright, .98, axis=0) - np.quantile(upright, .02, axis=0))
    scale = 1.3 / extent
    model.transform(pycolmap.Sim3d(scale, pycolmap.Rotation3d(rotation), -scale * rotation @ center))
    model.write(str(dataset / "sparse"))
    write_viewer_cameras(model, root)
    model.export_PLY(str(root / "subject-points.ply"))
    report = {"registered": model.num_reg_images(), "subject_points": int(keep.sum()),
              "removed_background_points": int((~keep).sum()), "scale": scale,
              "method": "Measured COLMAP poses; multi-view foreground consistency; no bottle shape prior"}
    write_json(root / "dataset-report.json", report)
    progress(root, "dataset-ready", **report)


def write_viewer_cameras(model, root):
    """Export calibrated poses for independent browser-renderer verification."""
    flip = np.diag([1., -1., -1.])
    poses = []
    for image in sorted(model.images.values(), key=lambda im: im.name):
        camera = model.cameras[image.camera_id]
        c2w = image.cam_from_world().inverse().matrix()
        eye = flip @ c2w[:, 3]
        forward = flip @ c2w[:, 2]
        up = -flip @ c2w[:, 1]
        k = camera.calibration_matrix()
        poses.append({'name': image.name, 'eye': eye.tolist(), 'target': (eye + forward).tolist(),
                      'up': up.tolist(), 'width': camera.width, 'height': camera.height,
                      'fov': math.degrees(2 * math.atan(camera.height / (2 * k[1, 1]))),
                      'intrinsics': k.tolist()})
    write_json(root / 'viewer-cameras.json', poses)


def train(args, root):
    if not args.brush.is_file():
        raise ValueError('Brush is missing. Run scripts/setup-reconstruction.ps1 first, or pass --brush.')
    if not (root / 'dataset/sparse/cameras.bin').is_file():
        raise ValueError('Measured dataset is missing; run extract,sfm,segment,prepare first.')
    output = root / args.training_name
    output.mkdir(exist_ok=True)
    if list(output.glob('*.ply')):
        raise ValueError('Training exports already exist; choose a new --training-name.')
    command = [str(args.brush.resolve()), str(root / 'dataset'), '--total-steps', str(args.steps),
               '--max-splats', str(args.max_splats), '--sh-degree', '3', '--refine-every', '100',
               '--growth-stop-iter', str(round(args.steps * .75)), '--max-resolution', str(args.max_dimension or 16384),
               '--eval-split-every', '8', '--eval-every', str(min(args.steps, args.checkpoint_every)), '--eval-save-to-disk',
               '--export-every', str(min(args.steps, args.checkpoint_every)), '--export-path', str(output)]
    write_json(output / 'training-config.json', {'command': command, 'held_out_every': 8, 'seed': 42})
    progress(root, 'training', steps=args.steps, log=str(output / 'train.log'))
    env = {**os.environ, 'RUST_LOG': 'warn,brush_cli=info,brush_process::eval_export=info'}
    with (output / 'train.log').open('w', encoding='utf-8') as log:
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    exports = sorted(output.glob('*.ply'))
    if not exports:
        raise ValueError('Training ended without exporting a model; inspect train.log')
    progress(root, 'trained', model=str(exports[-1]), next='Import the original PLY into the web viewer; run evaluate_reconstruction.py for held-out comparisons.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--max-dimension", type=int, default=1920, help="0 keeps native source resolution")
    parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 4))
    parser.add_argument("--stages", default="extract,sfm,segment,prepare,train,evaluate")
    parser.add_argument("--exhaustive", action="store_true", help="Match all frame pairs to close loops and bridge difficult portions")
    parser.add_argument("--segmentation-model", type=Path, default=Path(__file__).resolve().parents[1] / "public/models/u2netp.onnx")
    parser.add_argument('--brush', type=Path, default=Path(__file__).resolve().parents[1] / 'artifacts/tools/brush/brush_app.exe')
    parser.add_argument('--steps', type=int, default=16000)
    parser.add_argument('--max-splats', type=int, default=400000)
    parser.add_argument('--checkpoint-every', type=int, default=2000)
    parser.add_argument('--training-name', default='trained', help='A new subdirectory for each training run (no overwriting)')
    args = parser.parse_args()
    if not args.video.is_file() or not 16 <= args.frames <= 600:
        parser.error("Provide an existing video and between 16 and 600 target frames")
    if args.max_dimension < 0 or args.threads < 1 or args.steps < 1 or args.max_splats < 32 or args.checkpoint_every < 1:
        parser.error('Invalid resolution, thread count or training limits')
    if Path(args.training_name).name != args.training_name or args.training_name in ('.', '..'):
        parser.error('--training-name must be a single subdirectory name')
    stages = args.stages.split(',')
    if set(stages) - {'extract', 'sfm', 'segment', 'prepare', 'train', 'evaluate'}:
        parser.error('Unknown stage')
    root = args.output.resolve()
    repository = Path(__file__).resolve().parents[1]
    if root.is_relative_to(repository) and not root.is_relative_to(repository / 'artifacts'):
        parser.error('Keep private reconstruction outputs under ignored artifacts/ or outside the repository')
    root.mkdir(exist_ok=True, parents=True)
    # A resumed stage must still belong to this exact private source/configuration.
    if (root / 'frames.json').exists():
        extract(args, root)
    def evaluate_stage(args, root):
        from evaluate_reconstruction import evaluate
        evaluate(root, args.training_name)
        progress(root, 'complete', model=str(root / args.training_name / 'validation/best.ply'))
    actions = {'extract': extract, 'sfm': sfm, 'segment': segment, 'prepare': prepare,
               'train': train, 'evaluate': evaluate_stage}
    for stage in stages:
        try:
            actions[stage](args, root)
        except Exception as error:
            progress(root, 'failed', failed_stage=stage, error=str(error))
            raise


if __name__ == "__main__":
    main()
