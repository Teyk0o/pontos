"""Pytest configuration and shared fixtures."""
import os
import sys
import pytest
from pathlib import Path
import numpy as np
from PIL import Image


@pytest.fixture(autouse=True, scope="session")
def isolate_env():
    """Isolate .env file during tests."""
    # Clear loaded env vars before tests
    # Don't load .env for tests
    os.environ.pop("DOTENV_LOADED", None)


@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch):
    """Mock environment variables for testing."""
    # Clear all pontos-related env vars first
    for key in list(os.environ.keys()):
        if key.startswith("SH_") or key in ["DEVICE", "MODEL_PATH", "CONFIDENCE_THRESHOLD"]:
            monkeypatch.delenv(key, raising=False)

    # Set test defaults
    monkeypatch.setenv("SH_CLIENT_ID", "test-client-id")
    monkeypatch.setenv("SH_CLIENT_SECRET", "test-client-secret")
    monkeypatch.setenv("DEVICE", "cpu")
    monkeypatch.setenv("MODEL_PATH", "models/yolo11s_tci.pt")


@pytest.fixture(scope="session")
def test_data_dir():
    """Test data directory."""
    return Path(__file__).parent / "test_data"


@pytest.fixture(scope="session")
def sample_image(tmp_path_factory):
    """Create a sample RGB image for testing."""
    img_dir = tmp_path_factory.mktemp("images")
    img_path = img_dir / "test_image.png"

    # Create 1024x1024 RGB image with random data
    img_array = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
    Image.fromarray(img_array).save(img_path)

    return img_path


@pytest.fixture
def mock_sentinel_response():
    """Mock Sentinel Hub API response."""
    # Create mock 1024x1024 RGB image
    return np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)


@pytest.fixture
def sample_detections():
    """Sample YOLO detection results."""
    return [
        {
            "bbox": [100.0, 200.0, 150.0, 250.0],
            "confidence": 0.58,
            "class": "vessel",
            "center": [125.0, 225.0]
        },
        {
            "bbox": [300.0, 400.0, 350.0, 450.0],
            "confidence": 0.41,
            "class": "vessel",
            "center": [325.0, 425.0]
        }
    ]


@pytest.fixture
def toulon_bbox():
    """Toulon bounding box coordinates."""
    return (5.85, 43.08, 6.05, 43.18)


@pytest.fixture(scope="session")
def create_toulon_symlink():
    """Create symlink to toulon image in test-friendly location."""
    test_data = Path("tests/test_data")
    test_data.mkdir(exist_ok=True)

    # Find toulon image
    sources = [
        Path("data/samples/toulon_l1c.png"),
        Path("data/toulon_l1c.png"),
        Path("data/toulon_sentinel2.tiff"),
        Path("runs/toulon/predict/toulon_sentinel2.jpg")
    ]

    for src in sources:
        if src.exists():
            dst = test_data / "toulon_test.png"
            if not dst.exists():
                # Copy instead of symlink for cross-platform
                import shutil
                shutil.copy(src, dst)
            return dst

    return None

@pytest.fixture(scope="session")
def toulon_image(create_toulon_symlink):
    """Real Toulon image if available."""
    return create_toulon_symlink