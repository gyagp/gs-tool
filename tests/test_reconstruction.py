"""Run with artifacts/tools/reconstruction-python/Scripts/python.exe -m unittest discover -s tests -p test_reconstruction.py."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from reconstruct_video import extract, write_viewer_cameras
from evaluate_reconstruction import metrics


class ReconstructionRegression(unittest.TestCase):
    def video(self, folder, stationary=False):
        path = folder / 'source.avi'
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 60, (96, 160))
        self.assertTrue(writer.isOpened())
        rng = np.random.default_rng(42)
        image = rng.integers(30, 220, (160, 96, 3), dtype=np.uint8)
        for _ in range(120):
            if not stationary:
                image = rng.integers(30, 220, (160, 96, 3), dtype=np.uint8)
            writer.write(image)
        writer.release()
        return path

    def test_native_aspect_and_cache_configuration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = SimpleNamespace(video=self.video(root), frames=16, max_dimension=80)
            manifest = extract(args, root)
            self.assertEqual(len(manifest['frames']), 16)
            frame = cv2.imread(str(root / 'frames' / manifest['frames'][0]['name']))
            self.assertEqual(frame.shape[:2], (80, 48))
            self.assertEqual(extract(args, root), manifest)
            args.max_dimension = 160
            with self.assertRaisesRegex(ValueError, 'different video/configuration'):
                extract(args, root)

    def test_stationary_video_does_not_invent_new_views(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = SimpleNamespace(video=self.video(root, True), frames=16, max_dimension=0)
            with self.assertRaisesRegex(ValueError, 'Too few distinct'):
                extract(args, root)
            self.assertFalse((root / 'frames.json').exists())

    def test_calibrated_camera_flip_preserves_projection_including_roll(self):
        rotation = Rotation.from_euler('xyz', [19, -37, 12], degrees=True).as_matrix()
        center = np.array([.8, -.3, 1.7])
        c2w = np.column_stack((rotation, center))
        image = SimpleNamespace(name='held-out.png', camera_id=1,
            cam_from_world=lambda: SimpleNamespace(inverse=lambda: SimpleNamespace(matrix=lambda: c2w)))
        k = np.array([[700., 0, 320], [0, 700, 240], [0, 0, 1]])
        model = SimpleNamespace(images={1: image}, cameras={1: SimpleNamespace(width=640, height=480, calibration_matrix=lambda: k)})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_viewer_cameras(model, root)
            pose = json.loads((root / 'viewer-cameras.json').read_text())[0]
        eye, target, up = (np.array(pose[key]) for key in ('eye', 'target', 'up'))
        forward = target - eye
        right = np.cross(forward, up)
        point_camera = np.array([.24, -.1, 2.3])
        point_viewer = np.diag([1, -1, -1]) @ (rotation @ point_camera + center)
        relative = point_viewer - eye
        projected = np.array([relative @ right, -relative @ up, relative @ forward])
        np.testing.assert_allclose(projected, point_camera, atol=1e-12)
        self.assertAlmostEqual(240 / np.tan(np.deg2rad(pose['fov']) / 2), 700)

    def test_foreground_metrics_do_not_reward_empty_background(self):
        reference = np.zeros((64, 64, 3), np.uint8)
        reference[16:48, 16:48] = 100
        prediction = reference.copy()
        prediction[:12] = 255
        mask = np.zeros((64, 64), bool)
        mask[24:40, 24:40] = True
        result = metrics(reference, prediction, mask)
        self.assertGreater(result['foreground_psnr'], 100)
        self.assertGreater(result['foreground_ssim'], .999)
        self.assertLess(result['full_image_psnr'], 10)
        prediction[24:40, 24:40] = 0
        self.assertLess(metrics(reference, prediction, mask)['foreground_psnr'], 10)


if __name__ == '__main__':
    unittest.main()
