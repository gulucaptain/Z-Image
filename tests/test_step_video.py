"""Check the actual MP4 encoding and chronological frame order."""
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from zimage_app.step_video import create_step_video


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg/ffprobe required")
class StepVideoTests(unittest.TestCase):
    def test_browser_codec_timing_and_frame_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            for index, color in enumerate(["red", "green", "blue"], 1):
                Image.new("RGB", (321, 241), color).save(Path(tmp) / f"step_{index:03d}.png")
            video = create_step_video(tmp, 3)
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_streams", "-of", "json", video],
                check=True, capture_output=True, text=True,
            )
            stream = json.loads(probe.stdout)["streams"][0]
            self.assertEqual(stream["codec_name"], "h264")
            self.assertEqual(stream["pix_fmt"], "yuv420p")
            self.assertEqual(int(stream["nb_frames"]), 3)
            self.assertAlmostEqual(float(stream["duration"]), 1.5)
            frames = subprocess.run(
                ["ffmpeg", "-v", "error", "-i", video, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                check=True, capture_output=True,
            ).stdout
            frame_size = stream["width"] * stream["height"] * 3
            for index in range(3):
                # Check image content away from the label and padded edge.
                offset = index * frame_size + (100 * stream["width"] + 200) * 3
                pixel = frames[offset:offset + 3]
                self.assertEqual(max(range(3), key=lambda channel: pixel[channel]), index)

    def test_step_labels_change_on_identical_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            for index in (1, 2):
                Image.new("RGB", (320, 240), "gray").save(Path(tmp) / f"step_{index:03d}.png")
            video = create_step_video(tmp, 2)
            decoded = subprocess.run(
                ["ffmpeg", "-v", "error", "-i", video, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                check=True, capture_output=True,
            ).stdout
            size = 320 * 240 * 3
            first = Image.frombytes("RGB", (320, 240), decoded[:size])
            second = Image.frombytes("RGB", (320, 240), decoded[size:2 * size])
            label = first.crop((5, 5, 110, 40))
            self.assertGreater(label.getextrema()[0][1], 200)  # White text is visible.
            self.assertLess(label.getextrema()[0][0], 80)  # Dark backing is visible.
            self.assertNotEqual(label.tobytes(), second.crop((5, 5, 110, 40)).tobytes())
            # Source previews remain unmodified.
            self.assertEqual(Image.open(Path(tmp) / "step_001.png").getextrema(), ((128, 128),) * 3)


if __name__ == "__main__":
    unittest.main()
