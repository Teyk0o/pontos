"""Tests for vessel detector module."""

import pytest
import torch
from pathlib import Path
from unittest.mock import patch
from pontos.detector import VesselDetector


def test_detector_initialization():
    """Test detector initialization with default config."""
    detector = VesselDetector()

    assert detector.confidence_threshold == 0.05
    assert detector.model is not None
    assert detector.device in ["cpu", "0"]


def test_detector_cpu_fallback():
    """Test CPU fallback when GPU not available."""
    with patch("torch.cuda.is_available", return_value=False):
        detector = VesselDetector(device="0")
        assert detector.device == "cpu"


def test_detector_gpu_available():
    """Test GPU usage when available."""
    if torch.cuda.is_available():
        detector = VesselDetector(device="0")
        assert detector.device == "0"
        assert detector.is_gpu_available


def test_detector_custom_threshold():
    """Test custom confidence threshold."""
    detector = VesselDetector(confidence_threshold=0.25)
    assert detector.confidence_threshold == 0.25


@pytest.mark.skipif(
    not Path("models/yolo11s_tci.pt").exists(), reason="YOLO model not found"
)
def test_detect_sample_image(sample_image):
    """Test detection on sample image."""
    detector = VesselDetector(confidence_threshold=0.01)
    detections = detector.detect(sample_image)

    assert isinstance(detections, list)
    for det in detections:
        assert "bbox" in det
        assert "confidence" in det
        assert "class" in det
        assert "center" in det
        assert len(det["bbox"]) == 4
        assert 0 <= det["confidence"] <= 1


@pytest.mark.skipif(
    not Path("data/samples/toulon_l1c.png").exists(),
    reason="Toulon test image not found",
)
def test_detect_toulon_image(toulon_image):
    """Test detection on real Toulon image."""
    if toulon_image is None:
        pytest.skip("No Toulon image available")

    detector = VesselDetector(confidence_threshold=0.05)
    detections = detector.detect(toulon_image)

    # Toulon should have at least 5 vessels
    assert len(detections) >= 5

    # Check detection quality
    assert all(0 <= d["confidence"] <= 1 for d in detections)
    assert any(
        d["confidence"] > 0.3 for d in detections
    )  # At least one high-confidence


def test_detect_with_visualization(sample_image, tmp_path):
    """Test detection with visualization saved."""
    detector = VesselDetector()
    detections = detector.detect(
        sample_image, save_visualization=True, output_dir=tmp_path / "test_results"
    )

    assert isinstance(detections, list)
    # Check that output directory was created
    assert (tmp_path / "test_results").exists()


def test_get_device_name():
    """Test device name retrieval."""
    detector = VesselDetector()
    device_name = detector.get_device_name()

    assert isinstance(device_name, str)
    assert len(device_name) > 0


@pytest.mark.skipif(
    not Path("models/yolo11s_tci.pt").exists(), reason="YOLO model not found"
)
def test_detect_no_detections(tmp_path):
    """Test detection when no vessels found."""
    # Create blank image
    from PIL import Image
    import numpy as np

    blank_img = tmp_path / "blank.png"
    blank_array = np.ones((1024, 1024, 3), dtype=np.uint8) * 255  # All white
    Image.fromarray(blank_array).save(blank_img)

    detector = VesselDetector(confidence_threshold=0.9)  # Very high threshold
    detections = detector.detect(blank_img)

    # Might have 0 detections
    assert isinstance(detections, list)


def test_detector_model_path_custom(tmp_path):
    """Test detector with custom model path."""
    # This will fail if model doesn't exist, which is expected
    fake_model = tmp_path / "fake_model.pt"

    with pytest.raises(Exception):  # YOLO will raise if model not found
        VesselDetector(model_path=fake_model)


def test_detector_device_name_cpu():
    """Test device name when using CPU."""
    detector = VesselDetector(device="cpu")
    device_name = detector.get_device_name()

    assert device_name == "CPU"
    assert not detector.is_gpu_available


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
def test_detector_device_name_gpu():
    """Test device name when using GPU."""
    detector = VesselDetector(device="0")
    device_name = detector.get_device_name()

    assert device_name != "CPU"
    assert detector.is_gpu_available
    assert "Radeon" in device_name or "NVIDIA" in device_name or "AMD" in device_name


def test_detect_invalid_image_path():
    """Test detection with non-existent image."""
    detector = VesselDetector()

    # YOLO should handle this gracefully or raise
    with pytest.raises(Exception):
        detector.detect(Path("nonexistent_image.png"))


@pytest.mark.skipif(
    not Path("models/yolo11s_tci.pt").exists(), reason="YOLO model not found"
)
def test_detect_visualization_output_dir(sample_image, tmp_path):
    """Test that visualization is saved to correct directory."""
    detector = VesselDetector()

    output_dir = tmp_path / "custom_output"
    _ = detector.detect(sample_image, save_visualization=True, output_dir=output_dir)

    # Check output directory was created
    assert output_dir.exists()
