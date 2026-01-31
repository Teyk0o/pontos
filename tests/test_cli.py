"""Tests for command-line interface."""

import pytest
from click.testing import CliRunner
from unittest.mock import patch, MagicMock
from pontos.cli import cli


@pytest.fixture
def cli_runner():
    """Click CLI test runner."""
    return CliRunner()


def test_cli_help(cli_runner):
    """Test CLI help command."""
    result = cli_runner.invoke(cli, ["--help"])

    assert result.exit_code == 0
    assert "Pontos" in result.output
    assert "naval surveillance" in result.output.lower()


def test_scan_help(cli_runner):
    """Test scan command help."""
    result = cli_runner.invoke(cli, ["scan", "--help"])

    assert result.exit_code == 0
    assert "bbox" in result.output.lower()
    assert "date" in result.output.lower()


@patch("pontos.cli.SentinelDataSource")
@patch("pontos.cli.VesselDetector")
@patch("pontos.cli.GeoExporter")
def test_scan_command_success(
    mock_exporter, mock_detector, mock_sentinel, cli_runner, tmp_path
):
    """Test successful scan command."""
    # Mock Sentinel download
    mock_sentinel_instance = MagicMock()
    scene_path = tmp_path / "test_scene.png"
    scene_path.touch()
    mock_sentinel_instance.get_scene.return_value = scene_path
    mock_sentinel.return_value = mock_sentinel_instance

    # Mock detector
    mock_detector_instance = MagicMock()
    mock_detector_instance.detect.return_value = [
        {"bbox": [100, 100, 200, 200], "confidence": 0.8, "center": [150, 150]}
    ]
    mock_detector.return_value = mock_detector_instance

    # Mock exporter
    output_path = tmp_path / "output.geojson"
    mock_exporter.detections_to_geojson.return_value = output_path

    result = cli_runner.invoke(
        cli,
        [
            "scan",
            "--bbox",
            "5.85,43.08,6.05,43.18",
            "--date-start",
            "2026-01-01",
            "--date-end",
            "2026-01-31",
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    assert "Scanning" in result.output
    assert "Found" in result.output
    assert "vessels" in result.output.lower()


def test_scan_invalid_bbox(cli_runner):
    """Test scan with invalid bbox format."""
    result = cli_runner.invoke(
        cli,
        [
            "scan",
            "--bbox",
            "invalid",
            "--date-start",
            "2026-01-01",
            "--date-end",
            "2026-01-31",
        ],
    )

    assert result.exit_code != 0


def test_scan_missing_required_args(cli_runner):
    """Test scan without required arguments."""
    result = cli_runner.invoke(cli, ["scan"])

    assert result.exit_code != 0
    assert "Missing option" in result.output or "required" in result.output.lower()


@patch("pontos.cli.SentinelDataSource")
@patch("pontos.cli.VesselDetector")
def test_scan_custom_confidence(mock_detector, mock_sentinel, cli_runner, tmp_path):
    """Test scan with custom confidence threshold."""
    # Mock setup
    mock_sentinel_instance = MagicMock()
    scene_path = tmp_path / "test_scene.png"
    scene_path.touch()
    mock_sentinel_instance.get_scene.return_value = scene_path
    mock_sentinel.return_value = mock_sentinel_instance

    mock_detector_instance = MagicMock()
    mock_detector_instance.detect.return_value = []
    mock_detector.return_value = mock_detector_instance

    result = cli_runner.invoke(
        cli,
        [
            "scan",
            "--bbox",
            "5.85,43.08,6.05,43.18",
            "--date-start",
            "2026-01-01",
            "--date-end",
            "2026-01-31",
            "--conf",
            "0.25",
        ],
    )

    # Verify detector was called with custom threshold
    mock_detector.assert_called_with(confidence_threshold=0.25)
    assert result.exit_code == 0
