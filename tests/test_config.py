"""Tests for configuration module."""

import pytest
from pontos.config import PontosConfig


def test_config_from_env(monkeypatch):
    """Test configuration loading from environment."""
    monkeypatch.setenv("SH_CLIENT_ID", "test-id-override")
    monkeypatch.setenv("SH_CLIENT_SECRET", "test-secret-override")
    monkeypatch.setenv("DEVICE", "cpu")
    monkeypatch.setenv("CONFIDENCE_THRESHOLD", "0.1")

    # Create config after env is set
    config = PontosConfig()

    assert config.sentinel_client_id == "test-id-override"
    assert config.sentinel_client_secret == "test-secret-override"
    assert config.device == "cpu"
    assert config.confidence_threshold == 0.1


def test_config_all_env_vars(monkeypatch):
    """Test all environment variables are loaded."""
    monkeypatch.setenv("SH_CLIENT_ID", "test-id")
    monkeypatch.setenv("SH_CLIENT_SECRET", "test-secret")
    monkeypatch.setenv("DEVICE", "cpu")
    monkeypatch.setenv("CONFIDENCE_THRESHOLD", "0.25")
    monkeypatch.setenv("PATCH_SIZE", "640")
    monkeypatch.setenv("PATCH_OVERLAP", "0.75")
    monkeypatch.setenv("MAX_WORKERS", "8")
    monkeypatch.setenv("BATCH_SIZE", "16")

    config = PontosConfig()

    assert config.confidence_threshold == 0.25
    assert config.patch_size == 640
    assert config.patch_overlap == 0.75
    assert config.max_workers == 8
    assert config.batch_size == 16


def test_config_defaults(monkeypatch):
    """Test default configuration values."""
    # Clear env vars to test defaults
    monkeypatch.delenv("PATCH_SIZE", raising=False)
    monkeypatch.delenv("PATCH_OVERLAP", raising=False)
    monkeypatch.delenv("MAX_WORKERS", raising=False)

    config = PontosConfig()

    assert config.patch_size == 320
    assert config.patch_overlap == 0.5
    assert config.max_workers == 4
    assert config.batch_size == 8


def test_config_validation_missing_credentials(monkeypatch):
    """Test validation fails with missing credentials."""
    monkeypatch.delenv("SH_CLIENT_ID", raising=False)
    monkeypatch.delenv("SH_CLIENT_SECRET", raising=False)

    config = PontosConfig()

    # Validation must be called explicitly
    with pytest.raises(ValueError, match="Sentinel Hub credentials not configured"):
        config.validate()


def test_config_validation_missing_model(monkeypatch, tmp_path):
    """Test validation fails with missing model file."""
    monkeypatch.setenv("SH_CLIENT_ID", "test-id")
    monkeypatch.setenv("SH_CLIENT_SECRET", "test-secret")
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "nonexistent.pt"))

    config = PontosConfig()

    # Validation must be called explicitly
    with pytest.raises(FileNotFoundError, match="Model not found"):
        config.validate()


def test_config_validation_success(monkeypatch):
    """Test validation passes with valid config."""
    monkeypatch.setenv("SH_CLIENT_ID", "test-id")
    monkeypatch.setenv("SH_CLIENT_SECRET", "test-secret")
    monkeypatch.setenv("MODEL_PATH", "models/yolo11s_tci.pt")

    config = PontosConfig()

    # Should not raise if model exists and credentials set
    if config.model_path.exists():
        config.validate()  # Should pass
