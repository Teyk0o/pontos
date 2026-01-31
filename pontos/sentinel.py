"""Sentinel-2 L1C data acquisition via Sentinel Hub API."""

from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
from PIL import Image
from sentinelhub import (
    SHConfig,
    SentinelHubRequest,
    DataCollection,
    BBox,
    CRS,
    MimeType,
    MosaickingOrder,
)

from pontos.config import config


class SentinelDataSource:
    """Sentinel-2 L1C data acquisition client."""

    def __init__(
        self, client_id: Optional[str] = None, client_secret: Optional[str] = None
    ):
        """
        Initialize Sentinel Hub client.

        Args:
            client_id: Sentinel Hub OAuth client ID
            client_secret: Sentinel Hub OAuth client secret
        """
        self.sh_config = SHConfig()

        # Use provided credentials, or fallback to config
        if client_id is not None:
            self.sh_config.sh_client_id = client_id
        else:
            self.sh_config.sh_client_id = config.sentinel_client_id

        if client_secret is not None:
            self.sh_config.sh_client_secret = client_secret
        else:
            self.sh_config.sh_client_secret = config.sentinel_client_secret

        # Validate credentials
        if not self.sh_config.sh_client_id or not self.sh_config.sh_client_secret:
            raise ValueError("Sentinel Hub credentials not configured")

    def get_scene(
        self,
        bbox: Tuple[float, float, float, float],
        time_range: Tuple[str, str],
        size: int = 1024,  # Fixed size like prototype
        max_cloud_coverage: float = 0.2,
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        Download Sentinel-2 L1C RGB scene (Top of Atmosphere).

        Args:
            bbox: Bounding box as (min_lon, min_lat, max_lon, max_lat) in WGS84
            time_range: Time interval as (start_date, end_date) in ISO format
            size: Image size in pixels (square image)
            max_cloud_coverage: Maximum cloud coverage ratio (0.0 to 1.0)
            output_path: Path to save output image

        Returns:
            Path to saved PNG file
        """
        bbox_obj = BBox(bbox=bbox, crs=CRS.WGS84)

        # Evalscript for L1C RGB (exact same as prototype)
        evalscript = """
        // L1C TCI RGB (comme yolo11s_tci)
        return [B04, B03, B02];
        """

        request = SentinelHubRequest(
            evalscript=evalscript,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L1C,
                    time_interval=time_range,
                    mosaicking_order=MosaickingOrder.LEAST_CC,
                    maxcc=max_cloud_coverage,
                )
            ],
            responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
            bbox=bbox_obj,
            size=[size, size],  # Fixed square size
            config=self.sh_config,
        )

        # Execute request
        image_data = request.get_data()
        image_rgb = np.array(image_data[0])  # PNG already uint8, no scaling needed!

        # Save to disk
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = config.data_dir / f"sentinel2_l1c_{timestamp}.png"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image_rgb).save(output_path)

        return output_path
