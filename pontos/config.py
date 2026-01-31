"""Configuration management for Pontos ship detection system."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

if not os.getenv("PYTEST_CURRENT_TEST"):
    load_dotenv()


@dataclass
class PontosConfig:
    """Main configuration for Pontos."""

    # Paths
    model_path: Path = None
    data_dir: Path = Path("data")
    output_dir: Path = Path("runs")

    # Sentinel Hub API
    sentinel_client_id: Optional[str] = None
    sentinel_client_secret: Optional[str] = None

    # Detection parameters
    confidence_threshold: float = 0.05
    patch_size: int = 320
    patch_overlap: float = 0.5
    device: str = "0"

    # Processing
    max_workers: int = 4
    batch_size: int = 8

    def __post_init__(self):
        """Load values from environment after initialization."""
        self.model_path = Path(os.getenv("MODEL_PATH", "models/yolo11s_tci.pt"))
        self.sentinel_client_id = os.getenv("SH_CLIENT_ID")
        self.sentinel_client_secret = os.getenv("SH_CLIENT_SECRET")
        self.confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.05"))
        self.device = os.getenv("DEVICE", "0")
        self.patch_size = int(os.getenv("PATCH_SIZE", "320"))
        self.patch_overlap = float(os.getenv("PATCH_OVERLAP", "0.5"))
        self.max_workers = int(os.getenv("MAX_WORKERS", "4"))
        self.batch_size = int(os.getenv("BATCH_SIZE", "8"))

    def validate(self) -> None:
        """Validate configuration."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        if not self.sentinel_client_id or not self.sentinel_client_secret:
            raise ValueError(
                "Sentinel Hub credentials not configured.\n"
                "Please create a .env file with SH_CLIENT_ID and SH_CLIENT_SECRET.\n"
                "Get credentials at: https://apps.sentinel-hub.com/dashboard/#/account/settings"
            )


# Global config instance
config = PontosConfig()
