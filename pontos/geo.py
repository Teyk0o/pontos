"""Geospatial utilities for coordinate transformations and exports."""

import json
from pathlib import Path
from typing import List, Tuple


class GeoExporter:
    """Export detections to geospatial formats."""

    @staticmethod
    def detections_to_geojson(
        detections: List[dict],
        bbox: Tuple[float, float, float, float],
        image_size: Tuple[int, int],
        output_path: Path,
    ) -> Path:
        """
        Convert pixel-based detections to GeoJSON with geographic coordinates.

        Args:
            detections: List of detection dicts with 'bbox' and 'confidence'
            bbox: Geographic bounding box (min_lon, min_lat, max_lon, max_lat)
            image_size: Image dimensions (width, height) in pixels
            output_path: Path to save GeoJSON file

        Returns:
            Path to saved GeoJSON file
        """
        features = []

        for idx, detection in enumerate(detections):
            # Convert pixel coordinates to geographic coordinates
            cx_px, cy_px = detection["center"]
            lon, lat = GeoExporter._pixel_to_geo(cx_px, cy_px, bbox, image_size)

            feature = {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": {
                    "id": idx,
                    "confidence": detection["confidence"],
                    "class": detection.get("class", "vessel"),
                },
            }
            features.append(feature)

        geojson = {"type": "FeatureCollection", "features": features}

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(geojson, f, indent=2)

        return output_path

    @staticmethod
    def _pixel_to_geo(
        x_px: float,
        y_px: float,
        bbox: Tuple[float, float, float, float],
        image_size: Tuple[int, int],
    ) -> Tuple[float, float]:
        """
        Convert pixel coordinates to geographic coordinates.

        Args:
            x_px: X pixel coordinate
            y_px: Y pixel coordinate
            bbox: (min_lon, min_lat, max_lon, max_lat)
            image_size: (width, height) in pixels

        Returns:
            (longitude, latitude) in WGS84
        """
        min_lon, min_lat, max_lon, max_lat = bbox
        width, height = image_size

        lon = min_lon + (x_px / width) * (max_lon - min_lon)
        lat = max_lat - (y_px / height) * (max_lat - min_lat)

        return lon, lat
