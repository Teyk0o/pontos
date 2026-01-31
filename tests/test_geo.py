"""Tests for geospatial utilities."""

import json
from pontos.geo import GeoExporter


def test_pixel_to_geo_center(toulon_bbox):
    """Test pixel to geographic coordinate conversion (center)."""
    lon, lat = GeoExporter._pixel_to_geo(
        x_px=512,
        y_px=512,  # Center of 1024x1024
        bbox=toulon_bbox,
        image_size=(1024, 1024),
    )

    # Should be near center of bbox
    center_lon = (toulon_bbox[0] + toulon_bbox[2]) / 2
    center_lat = (toulon_bbox[1] + toulon_bbox[3]) / 2

    assert abs(lon - center_lon) < 0.01
    assert abs(lat - center_lat) < 0.01


def test_pixel_to_geo_corners(toulon_bbox):
    """Test pixel to geographic conversion at corners."""
    # Top-left corner
    lon, lat = GeoExporter._pixel_to_geo(0, 0, toulon_bbox, (1024, 1024))
    assert abs(lon - toulon_bbox[0]) < 0.01  # min_lon
    assert abs(lat - toulon_bbox[3]) < 0.01  # max_lat

    # Bottom-right corner
    lon, lat = GeoExporter._pixel_to_geo(1024, 1024, toulon_bbox, (1024, 1024))
    assert abs(lon - toulon_bbox[2]) < 0.01  # max_lon
    assert abs(lat - toulon_bbox[1]) < 0.01  # min_lat


def test_detections_to_geojson(sample_detections, toulon_bbox, tmp_path):
    """Test GeoJSON export from detections."""
    output_path = tmp_path / "test_vessels.geojson"

    result = GeoExporter.detections_to_geojson(
        detections=sample_detections,
        bbox=toulon_bbox,
        image_size=(1024, 1024),
        output_path=output_path,
    )

    assert result == output_path
    assert output_path.exists()

    # Validate GeoJSON structure
    with open(output_path) as f:
        geojson = json.load(f)

    assert geojson["type"] == "FeatureCollection"
    assert len(geojson["features"]) == len(sample_detections)

    for feature in geojson["features"]:
        assert feature["type"] == "Feature"
        assert feature["geometry"]["type"] == "Point"
        assert len(feature["geometry"]["coordinates"]) == 2
        assert "confidence" in feature["properties"]
        assert "id" in feature["properties"]


def test_geojson_coordinates_in_bbox(sample_detections, toulon_bbox, tmp_path):
    """Test that exported coordinates are within bbox."""
    output_path = tmp_path / "test_vessels.geojson"

    GeoExporter.detections_to_geojson(
        sample_detections, toulon_bbox, (1024, 1024), output_path
    )

    with open(output_path) as f:
        geojson = json.load(f)

    for feature in geojson["features"]:
        lon, lat = feature["geometry"]["coordinates"]

        assert toulon_bbox[0] <= lon <= toulon_bbox[2], f"Longitude {lon} out of bbox"
        assert toulon_bbox[1] <= lat <= toulon_bbox[3], f"Latitude {lat} out of bbox"
