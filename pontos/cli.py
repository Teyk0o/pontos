"""Command-line interface for Pontos."""

import click
from pathlib import Path
from pontos.detector import VesselDetector
from pontos.sentinel import SentinelDataSource
from pontos.geo import GeoExporter


@click.group()
def cli():
    """Pontos: Global naval surveillance."""
    pass


@cli.command()
@click.option(
    "--bbox", required=True, help="Bounding box: min_lon,min_lat,max_lon,max_lat"
)
@click.option("--date-start", required=True, help="Start date: YYYY-MM-DD")
@click.option("--date-end", required=True, help="End date: YYYY-MM-DD")
@click.option("--output", "-o", default="vessels.geojson", help="Output GeoJSON path")
@click.option("--conf", default=0.05, help="Confidence threshold")
def scan(bbox, date_start, date_end, output, conf):
    """Scan area of interest for vessels."""
    bbox_coords = tuple(map(float, bbox.split(",")))

    click.echo(f"Scanning {bbox_coords}...")

    # Download scene
    sentinel = SentinelDataSource()
    scene = sentinel.get_scene(bbox_coords, (date_start, date_end))

    # Detect
    detector = VesselDetector(confidence_threshold=conf)
    detections = detector.detect(scene)

    click.echo(f"Found {len(detections)} vessels")

    # Export
    GeoExporter.detections_to_geojson(
        detections, bbox_coords, (1024, 1024), Path(output)
    )
    click.echo(f"Saved: {output}")


if __name__ == "__main__":
    cli()
