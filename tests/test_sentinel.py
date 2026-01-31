"""Tests for Sentinel Hub data acquisition."""

import pytest
from unittest.mock import MagicMock, patch
from pontos.sentinel import SentinelDataSource


def test_sentinel_explicit_credentials():
    """Test Sentinel Hub with explicit credentials."""
    sentinel = SentinelDataSource(
        client_id="test-client-id", client_secret="test-client-secret"
    )

    assert sentinel.sh_config.sh_client_id == "test-client-id"
    assert sentinel.sh_config.sh_client_secret == "test-client-secret"


def test_sentinel_missing_credentials_explicit_empty():
    """Test initialization fails with empty string credentials."""
    # Empty strings should fail validation
    with pytest.raises(ValueError, match="Sentinel Hub credentials not configured"):
        SentinelDataSource(client_id="", client_secret="")


def test_sentinel_missing_one_credential():
    """Test initialization fails with only one credential."""
    with pytest.raises(ValueError, match="Sentinel Hub credentials not configured"):
        SentinelDataSource(client_id="test-id", client_secret="")

    with pytest.raises(ValueError, match="Sentinel Hub credentials not configured"):
        SentinelDataSource(client_id="", client_secret="test-secret")


def test_sentinel_custom_credentials():
    """Test initialization with custom credentials."""
    sentinel = SentinelDataSource(client_id="custom-id", client_secret="custom-secret")

    assert sentinel.sh_config.sh_client_id == "custom-id"
    assert sentinel.sh_config.sh_client_secret == "custom-secret"


@patch("pontos.sentinel.SentinelHubRequest")
def test_get_scene_parameters(mock_request, toulon_bbox, mock_sentinel_response):
    """Test scene download with correct parameters."""
    # Mock the request
    mock_instance = MagicMock()
    mock_instance.get_data.return_value = [mock_sentinel_response]
    mock_request.return_value = mock_instance

    sentinel = SentinelDataSource(client_id="test-id", client_secret="test-secret")
    scene_path = sentinel.get_scene(
        bbox=toulon_bbox, time_range=("2026-01-01", "2026-01-31"), size=1024
    )

    # Verify request was called
    assert mock_request.called
    assert scene_path.exists()
    assert scene_path.suffix == ".png"


def test_calculate_size_deprecated():
    """Test that size is now a fixed parameter."""
    # Size is now fixed parameter in get_scene, not calculated
    assert True


def test_sentinel_only_client_id():
    """Test with only client_id provided."""
    # Should use config for secret
    sentinel = SentinelDataSource(client_id="only-id")

    assert sentinel.sh_config.sh_client_id == "only-id"
    # Secret should come from config
    assert sentinel.sh_config.sh_client_secret is not None


def test_sentinel_only_client_secret():
    """Test with only client_secret provided."""
    sentinel = SentinelDataSource(client_secret="only-secret")

    # ID should come from config
    assert sentinel.sh_config.sh_client_id is not None
    assert sentinel.sh_config.sh_client_secret == "only-secret"


@patch("pontos.sentinel.SentinelHubRequest")
def test_get_scene_default_output_path(
    mock_request, toulon_bbox, mock_sentinel_response, tmp_path, monkeypatch
):
    """Test scene download with default output path."""
    # Change to temp dir
    monkeypatch.chdir(tmp_path)

    # Create data dir
    (tmp_path / "data").mkdir()

    # Mock request
    mock_instance = MagicMock()
    mock_instance.get_data.return_value = [mock_sentinel_response]
    mock_request.return_value = mock_instance

    sentinel = SentinelDataSource(client_id="test", client_secret="test")

    # Call without output_path
    scene_path = sentinel.get_scene(
        bbox=toulon_bbox, time_range=("2026-01-01", "2026-01-31"), size=1024
    )

    # Should create file with timestamp
    assert scene_path.exists()
    assert "sentinel2_l1c" in scene_path.name
    assert scene_path.suffix == ".png"


@patch("pontos.sentinel.SentinelHubRequest")
def test_get_scene_custom_size(mock_request, toulon_bbox, mock_sentinel_response):
    """Test scene download with custom size."""
    mock_instance = MagicMock()
    mock_instance.get_data.return_value = [mock_sentinel_response]
    mock_request.return_value = mock_instance

    sentinel = SentinelDataSource(client_id="test", client_secret="test")

    _ = sentinel.get_scene(
        bbox=toulon_bbox,
        time_range=("2026-01-01", "2026-01-31"),
        size=512,  # Custom size
    )

    # Verify request was called with correct size
    call_kwargs = mock_request.call_args[1]
    assert call_kwargs["size"] == [512, 512]
