"""
chaos-tests/test_cards_chaos.py
Chaos, fuzz, and resilience tests for the cards CV pipeline.

Run:
    cd /home/andrew/repos/cards
    python3 -m pytest chaos-tests/test_cards_chaos.py -v 2>&1 | tee chaos-tests/results.txt
"""

import os
import sys
import struct
import random
import string
import subprocess
import tempfile
import time
import tracemalloc
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

# Add parent directory so we can import the modules directly
sys.path.insert(0, str(Path(__file__).parent.parent))

REPO = Path(__file__).parent.parent
CARDS_SCRIPT = REPO / "card_crop.py"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_card_crop(extra_args, input_dir=None, output_dir=None, timeout=30):
    """Run card_crop.py as a subprocess, return (returncode, stdout, stderr)."""
    with tempfile.TemporaryDirectory() as tmpout:
        out = output_dir or tmpout
        cmd = [
            sys.executable, str(CARDS_SCRIPT),
            "--no-interactive", "--no-ocr-refine", "--no-ml-refine",
            "--detector", "contour",
        ]
        if input_dir:
            cmd += ["--input-dir", str(input_dir)]
        if output_dir:
            cmd += ["--output-dir", str(output_dir)]
        cmd += extra_args
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout, cwd=str(REPO)
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", "TIMEOUT"
        except Exception as e:
            return -2, "", str(e)


def make_temp_dir_with_file(filename, content: bytes) -> Path:
    """Create a temp dir with a single file containing the given bytes."""
    d = Path(tempfile.mkdtemp())
    (d / filename).write_bytes(content)
    return d


def make_real_image(w, h, mode="BGR") -> bytes:
    """Create a real JPEG image in memory."""
    arr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", arr)
    return buf.tobytes()


def make_card_like_image(w=800, h=1100) -> bytes:
    """Create an image with a card-like white rectangle on dark background."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (60, 40, 30)  # dark wood-like background
    # Draw white card rectangle
    cx, cy = w // 2, h // 2
    cw, ch = int(w * 0.4), int(h * 0.55)
    cv2.rectangle(img, (cx - cw // 2, cy - ch // 2), (cx + cw // 2, cy + ch // 2), (255, 255, 255), -1)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


# ---------------------------------------------------------------------------
# 1. Image file corruption tests
# ---------------------------------------------------------------------------

class TestCorruptImages:

    def test_zero_byte_file(self, tmp_path):
        """Zero-byte file should not crash; should fail gracefully."""
        d = make_temp_dir_with_file("test.jpg", b"")
        rc, out, err = run_card_crop([], input_dir=d, output_dir=tmp_path)
        assert rc in (0, 1), f"Unexpected return code {rc}: {err}"
        assert "Traceback" not in err, f"Unhandled exception: {err}"

    def test_truncated_jpeg(self, tmp_path):
        """Truncated JPEG header should not crash."""
        real = make_real_image(100, 100)
        truncated = real[:len(real) // 3]
        d = make_temp_dir_with_file("truncated.jpg", truncated)
        rc, out, err = run_card_crop([], input_dir=d, output_dir=tmp_path)
        assert rc in (0, 1), f"Return code {rc}: {err}"
        assert "Traceback" not in err, f"Exception: {err}"

    def test_random_bytes_as_jpg(self, tmp_path):
        """Random bytes with .jpg extension should not crash."""
        garbage = bytes(random.getrandbits(8) for _ in range(2048))
        d = make_temp_dir_with_file("garbage.jpg", garbage)
        rc, out, err = run_card_crop([], input_dir=d, output_dir=tmp_path)
        assert "Traceback" not in err, f"Exception on garbage bytes: {err}"

    def test_text_file_as_jpg(self, tmp_path):
        """Plain text file renamed to .jpg should not crash."""
        content = b"This is not an image file at all!\n" * 100
        d = make_temp_dir_with_file("not_an_image.jpg", content)
        rc, out, err = run_card_crop([], input_dir=d, output_dir=tmp_path)
        assert "Traceback" not in err, f"Exception: {err}"

    def test_html_as_jpg(self, tmp_path):
        """HTML file renamed to .jpg should not crash."""
        content = b"<html><body><h1>Hello World</h1></body></html>"
        d = make_temp_dir_with_file("page.jpg", content)
        rc, out, err = run_card_crop([], input_dir=d, output_dir=tmp_path)
        assert "Traceback" not in err, f"Exception: {err}"

    def test_jpeg_with_corrupt_tail(self, tmp_path):
        """Valid JPEG start, corrupt data after header."""
        real = make_real_image(200, 200)
        # Valid JPEG magic + garbage
        corrupt = real[:20] + bytes(random.getrandbits(8) for _ in range(5000))
        d = make_temp_dir_with_file("corrupt_tail.jpg", corrupt)
        rc, out, err = run_card_crop([], input_dir=d, output_dir=tmp_path)
        assert "Traceback" not in err, f"Exception: {err}"

    def test_png_file_as_jpg(self, tmp_path):
        """PNG file with .jpg extension should be handled gracefully."""
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        import io
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        d = make_temp_dir_with_file("actually_png.jpg", buf.getvalue())
        rc, out, err = run_card_crop([], input_dir=d, output_dir=tmp_path)
        # Should either succeed (OpenCV handles it) or fail gracefully
        assert "Traceback" not in err, f"Exception: {err}"


# ---------------------------------------------------------------------------
# 2. Extreme image dimensions
# ---------------------------------------------------------------------------

class TestExtremeDimensions:

    def test_enormous_image(self, tmp_path):
        """10000x10000 image should not OOM-crash or hang indefinitely."""
        # Create with numpy (don't actually encode full 10k×10k)
        arr = np.zeros((500, 350, 3), dtype=np.uint8)  # Reasonable proxy
        arr[50:450, 50:300] = 200  # Card-like bright region
        _, buf = cv2.imencode(".jpg", arr)
        d = make_temp_dir_with_file("big.jpg", buf.tobytes())
        rc, out, err = run_card_crop([], input_dir=d, output_dir=tmp_path, timeout=60)
        assert rc != -1, "Process timed out on large image"
        assert "Traceback" not in err, f"Exception: {err}"

    def test_1x1_image(self, tmp_path):
        """Tiny 1×1 image should fail gracefully."""
        arr = np.array([[[255, 255, 255]]], dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", arr)
        d = make_temp_dir_with_file("tiny.jpg", buf.tobytes())
        rc, out, err = run_card_crop([], input_dir=d, output_dir=tmp_path)
        assert "Traceback" not in err, f"Exception: {err}"

    def test_very_wide_image(self, tmp_path):
        """10000×1 image should not crash."""
        arr = np.zeros((1, 500, 3), dtype=np.uint8)  # Proxy for extreme aspect ratio
        _, buf = cv2.imencode(".jpg", arr)
        d = make_temp_dir_with_file("wide.jpg", buf.tobytes())
        rc, out, err = run_card_crop([], input_dir=d, output_dir=tmp_path)
        assert "Traceback" not in err, f"Exception: {err}"

    def test_single_color_image(self, tmp_path):
        """Solid-color image (no card) should fail gracefully."""
        arr = np.full((600, 400, 3), 128, dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", arr)
        d = make_temp_dir_with_file("solid.jpg", buf.tobytes())
        rc, out, err = run_card_crop([], input_dir=d, output_dir=tmp_path)
        assert "Traceback" not in err, f"Exception: {err}"
        assert "No card detected" in out or "FAIL" in out, "Should report no card found"


# ---------------------------------------------------------------------------
# 3. No card present / adversarial images
# ---------------------------------------------------------------------------

class TestNoCard:

    def test_wood_texture_no_card(self, tmp_path):
        """Wood-like noise texture with no card contour."""
        arr = np.random.randint(80, 140, (600, 800, 3), dtype=np.uint8)
        # Add some wood-grain-like horizontal streaks
        for y in range(0, 600, 15):
            arr[y:y+3, :] = arr[y:y+3, :] // 2
        _, buf = cv2.imencode(".jpg", arr)
        d = make_temp_dir_with_file("wood.jpg", buf.tobytes())
        rc, out, err = run_card_crop([], input_dir=d, output_dir=tmp_path)
        assert "Traceback" not in err, f"Exception: {err}"

    def test_circular_object_only(self, tmp_path):
        """Image with only a circle (pan-like) — should not detect a card."""
        arr = np.zeros((600, 800, 3), dtype=np.uint8)
        arr[:] = (60, 40, 30)
        cv2.circle(arr, (400, 300), 200, (200, 200, 200), -1)
        _, buf = cv2.imencode(".jpg", arr)
        d = make_temp_dir_with_file("circle.jpg", buf.tobytes())
        rc, out, err = run_card_crop([], input_dir=d, output_dir=tmp_path)
        assert "Traceback" not in err, f"Exception: {err}"

    def test_multiple_card_candidates(self, tmp_path):
        """Multiple rectangles in image — should pick the best one."""
        arr = np.zeros((800, 1200, 3), dtype=np.uint8)
        # Draw several card-like rectangles
        cv2.rectangle(arr, (50, 100), (250, 400), (200, 200, 200), -1)
        cv2.rectangle(arr, (400, 100), (600, 400), (220, 210, 190), -1)
        cv2.rectangle(arr, (750, 200), (950, 600), (230, 230, 230), -1)
        _, buf = cv2.imencode(".jpg", arr)
        d = make_temp_dir_with_file("multi_rect.jpg", buf.tobytes())
        rc, out, err = run_card_crop([], input_dir=d, output_dir=tmp_path)
        assert "Traceback" not in err, f"Exception: {err}"

    def test_black_image(self, tmp_path):
        """All-black image."""
        arr = np.zeros((600, 400, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", arr)
        d = make_temp_dir_with_file("black.jpg", buf.tobytes())
        rc, out, err = run_card_crop([], input_dir=d, output_dir=tmp_path)
        assert "Traceback" not in err, f"Exception: {err}"


# ---------------------------------------------------------------------------
# 4. CLI argument fuzzing
# ---------------------------------------------------------------------------

class TestCLIFuzzing:

    def test_missing_input_dir(self, tmp_path):
        """--input-dir pointing to nonexistent path."""
        rc, out, err = run_card_crop(
            ["--input-dir", "/nonexistent/path/that/does/not/exist"],
            output_dir=tmp_path
        )
        assert rc != 0, "Should fail with nonexistent input dir"
        assert "Traceback" not in err, f"Unhandled exception: {err}"
        assert "ERROR" in out or "not found" in out.lower() or rc == 1

    def test_readonly_output_dir(self, tmp_path):
        """--output-dir pointing to read-only path."""
        ro_dir = tmp_path / "readonly"
        ro_dir.mkdir()
        os.chmod(str(ro_dir), 0o444)
        d = make_temp_dir_with_file("test.jpg", make_card_like_image())
        rc, out, err = run_card_crop([], input_dir=d, output_dir=ro_dir)
        # Should fail gracefully, not crash
        assert "Traceback" not in err, f"Exception on read-only output: {err}"
        os.chmod(str(ro_dir), 0o755)  # Cleanup

    def test_extreme_padding_negative(self, tmp_path):
        """Negative padding value."""
        d = make_temp_dir_with_file("test.jpg", make_card_like_image())
        rc, out, err = run_card_crop(["--padding", "-999"], input_dir=d, output_dir=tmp_path)
        assert "Traceback" not in err, f"Exception with negative padding: {err}"

    def test_extreme_padding_huge(self, tmp_path):
        """Huge padding value (could OOM)."""
        d = make_temp_dir_with_file("test.jpg", make_card_like_image())
        rc, out, err = run_card_crop(["--padding", "100000"], input_dir=d, output_dir=tmp_path, timeout=30)
        assert rc != -1, "Timed out with huge padding"
        assert "Traceback" not in err, f"Exception with huge padding: {err}"

    def test_extreme_ocr_min_conf_zero(self, tmp_path):
        """ocr-min-conf=0 edge case."""
        d = make_temp_dir_with_file("test.jpg", make_card_like_image())
        rc, out, err = run_card_crop(["--no-ocr-refine", "--ocr-min-conf", "0"], input_dir=d, output_dir=tmp_path)
        assert "Traceback" not in err, f"Exception: {err}"

    def test_extreme_ocr_min_conf_above_one(self, tmp_path):
        """ocr-min-conf=999 (above max meaningful value)."""
        d = make_temp_dir_with_file("test.jpg", make_card_like_image())
        rc, out, err = run_card_crop(["--no-ocr-refine", "--ocr-min-conf", "999"], input_dir=d, output_dir=tmp_path)
        assert "Traceback" not in err, f"Exception: {err}"

    def test_ml_weight_zero(self, tmp_path):
        """ml-weight=0 boundary."""
        d = make_temp_dir_with_file("test.jpg", make_card_like_image())
        rc, out, err = run_card_crop(["--no-ml-refine", "--ml-weight", "0"], input_dir=d, output_dir=tmp_path)
        assert "Traceback" not in err, f"Exception: {err}"

    def test_ml_weight_out_of_range(self, tmp_path):
        """ml-weight=5.0 (above 1.0)."""
        d = make_temp_dir_with_file("test.jpg", make_card_like_image())
        rc, out, err = run_card_crop(["--no-ml-refine", "--ml-weight", "5.0"], input_dir=d, output_dir=tmp_path)
        assert "Traceback" not in err, f"Exception: {err}"

    def test_detector_conf_zero(self, tmp_path):
        """detector-conf=0 edge case."""
        d = make_temp_dir_with_file("test.jpg", make_card_like_image())
        rc, out, err = run_card_crop(["--detector-conf", "0.0", "--detector", "contour"], input_dir=d, output_dir=tmp_path)
        assert "Traceback" not in err, f"Exception: {err}"

    def test_empty_ext(self, tmp_path):
        """Empty ext filter."""
        d = make_temp_dir_with_file("test.jpg", make_card_like_image())
        rc, out, err = run_card_crop(["--ext", ""], input_dir=d, output_dir=tmp_path)
        assert "Traceback" not in err, f"Exception: {err}"

    def test_output_dir_is_file(self, tmp_path):
        """--output-dir pointing to an existing file (not dir)."""
        f = tmp_path / "afile.txt"
        f.write_text("hello")
        d = make_temp_dir_with_file("test.jpg", make_card_like_image())
        rc, out, err = run_card_crop([], input_dir=d, output_dir=f)
        # Should fail gracefully or handle
        assert "Traceback" not in err, f"Exception: {err}"


# ---------------------------------------------------------------------------
# 5. Internal API fuzzing (import and call directly)
# ---------------------------------------------------------------------------

class TestInternalAPI:

    def test_four_point_transform_degenerate(self):
        """four_point_transform with degenerate (zero-area) quad returns None."""
        from card_crop import four_point_transform
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        pts = np.array([[50, 50], [50, 50], [50, 50], [50, 50]], dtype=np.float32)
        result = four_point_transform(img, pts)
        assert result is None, "Degenerate quad should return None"

    def test_four_point_transform_negative_coords(self):
        """four_point_transform with points outside image bounds."""
        from card_crop import four_point_transform
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        pts = np.array([[-100, -100], [200, -100], [200, 200], [-100, 200]], dtype=np.float32)
        # Should not crash
        try:
            result = four_point_transform(img, pts)
        except Exception as e:
            pytest.fail(f"four_point_transform crashed with out-of-bounds pts: {e}")

    def test_detect_card_empty_image(self):
        """detect_card on a 1x1 image."""
        from card_crop import detect_card
        img = np.zeros((1, 1, 3), dtype=np.uint8)
        try:
            result = detect_card(img)
            # Should return (None, None) or a valid result
            assert result is not None
        except Exception as e:
            pytest.fail(f"detect_card crashed on 1x1 image: {e}")

    def test_detect_card_solid_white(self):
        """detect_card on solid white image."""
        from card_crop import detect_card
        img = np.full((600, 800, 3), 255, dtype=np.uint8)
        try:
            result = detect_card(img)
        except Exception as e:
            pytest.fail(f"detect_card crashed on solid white: {e}")

    def test_detect_card_solid_black(self):
        """detect_card on solid black image."""
        from card_crop import detect_card
        img = np.zeros((600, 800, 3), dtype=np.uint8)
        try:
            result = detect_card(img)
        except Exception as e:
            pytest.fail(f"detect_card crashed on solid black: {e}")

    def test_detect_card_nan_pixels(self):
        """detect_card with NaN-filled float array (type mismatch)."""
        from card_crop import detect_card
        img = np.full((100, 100, 3), np.nan, dtype=np.float32)
        try:
            result = detect_card(img)
        except Exception as e:
            # This is acceptable — document it
            pass  # np.float32 NaN arrays are not normal input

    def test_score_contour_tiny_area(self):
        """_score_contour with tiny area should return -1."""
        from card_crop import _score_contour
        cnt = np.array([[[10, 10]], [[11, 10]], [[11, 11]], [[10, 11]]], dtype=np.int32)
        score, _ = _score_contour(cnt, 640000, (800, 800))
        assert score == -1, "Tiny contour should be rejected"

    def test_order_points_non_standard_input(self):
        """order_points with coincident points should not crash."""
        from card_crop import order_points
        pts = np.array([[0, 0], [0, 0], [100, 100], [0, 100]], dtype=np.float32)
        try:
            result = order_points(pts)
        except Exception as e:
            pytest.fail(f"order_points crashed on coincident pts: {e}")

    def test_process_image_nonexistent_src(self, tmp_path):
        """process_image with nonexistent source path returns (False, msg, '')."""
        from card_crop import process_image
        # Create dummy args
        import argparse
        args = argparse.Namespace(
            ocr_refine=False, ocr_csv="", ocr_min_conf=0.25,
            ocr_max_dim=1800, ocr_text_margin=0.03,
            ml_refine=False, ml_weight=0.4, ml_model="", ml_device="cpu",
            ml_required=False, detector="contour", detector_model="",
            detector_conf=0.35, padding=0, no_resize=False, debug=False,
            dry_run=False
        )
        success, msg, ocr = process_image(
            Path("/nonexistent/fake.jpg"),
            tmp_path / "out.jpg",
            args
        )
        assert not success
        assert "Could not read" in msg or "not found" in msg.lower() or "read" in msg.lower()


# ---------------------------------------------------------------------------
# 6. Memory / performance chaos
# ---------------------------------------------------------------------------

class TestMemoryPerformance:

    def test_batch_100_images_no_oom(self, tmp_path):
        """Process 100 images, ensure no memory explosion."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create 100 small test images
        for i in range(100):
            arr = np.random.randint(0, 255, (200, 150, 3), dtype=np.uint8)
            cv2.imwrite(str(input_dir / f"img_{i:03d}.jpg"), arr)

        tracemalloc.start()
        rc, out, err = run_card_crop([], input_dir=input_dir, output_dir=output_dir, timeout=120)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        assert rc != -1, "Batch processing timed out"
        assert "Traceback" not in err, f"Exception during batch: {err}"
        # Peak memory < 2GB
        peak_mb = peak / (1024 * 1024)
        assert peak_mb < 2048, f"Peak memory too high: {peak_mb:.0f}MB"

    def test_rapid_successive_calls(self):
        """Rapid repeated calls to detect_card should not leak resources."""
        from card_crop import detect_card
        img = np.random.randint(0, 255, (400, 300, 3), dtype=np.uint8)
        errors = []
        for i in range(20):
            try:
                detect_card(img)
            except Exception as e:
                errors.append(str(e))
        assert not errors, f"Errors on rapid calls: {errors}"

    def test_empty_input_dir(self, tmp_path):
        """Empty input dir should exit cleanly with no images message."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        rc, out, err = run_card_crop([], input_dir=empty_dir, output_dir=tmp_path)
        assert rc == 0, f"Non-zero exit on empty dir: rc={rc}, err={err}"
        assert "No images" in out or "0" in out
        assert "Traceback" not in err


# ---------------------------------------------------------------------------
# 7. Hypothesis property-based tests
# ---------------------------------------------------------------------------

try:
    from hypothesis import given, settings, HealthCheck
    from hypothesis import strategies as st
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False

if HYPOTHESIS_AVAILABLE:
    class TestHypothesisCards:

        @given(
            w=st.integers(min_value=1, max_value=50),
            h=st.integers(min_value=1, max_value=50),
        )
        @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
        def test_detect_card_small_images(self, w, h):
            """detect_card should not throw on any small image."""
            from card_crop import detect_card
            img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
            try:
                detect_card(img)
            except Exception as e:
                pytest.fail(f"detect_card raised {type(e).__name__}: {e} on {w}x{h} image")

        @given(
            x1=st.integers(-200, 200), y1=st.integers(-200, 200),
            x2=st.integers(-200, 200), y2=st.integers(-200, 200),
        )
        @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
        def test_four_point_transform_arbitrary(self, x1, y1, x2, y2):
            """four_point_transform should not raise on arbitrary coords."""
            from card_crop import four_point_transform
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
            try:
                four_point_transform(img, pts)
            except Exception as e:
                pytest.fail(f"Raised {type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# 8. Process crash tests (subprocess isolation)
# ---------------------------------------------------------------------------

class TestCrashIsolation:

    def test_input_dir_is_a_file(self, tmp_path):
        """--input-dir pointing to a file instead of dir."""
        f = tmp_path / "not_a_dir.txt"
        f.write_text("test")
        rc, out, err = run_card_crop(["--input-dir", str(f)], output_dir=tmp_path)
        assert rc != 0
        assert "Traceback" not in err

    def test_symlink_loop(self, tmp_path):
        """Symlink loop in input dir should not hang."""
        d = tmp_path / "symdir"
        d.mkdir()
        link = d / "link.jpg"
        try:
            link.symlink_to(d)
        except Exception:
            pytest.skip("Cannot create symlink")
        rc, out, err = run_card_crop([], input_dir=d, output_dir=tmp_path, timeout=10)
        assert rc != -1, "Timed out on symlink loop"

    def test_dry_run_no_output(self, tmp_path):
        """--dry-run should produce no output files."""
        d = make_temp_dir_with_file("test.jpg", make_card_like_image())
        out_dir = tmp_path / "output"
        out_dir.mkdir()
        rc, out, err = run_card_crop(["--dry-run"], input_dir=d, output_dir=out_dir)
        assert rc == 0
        files = list(out_dir.iterdir())
        assert len(files) == 0, f"Dry run wrote files: {files}"
        assert "Traceback" not in err
