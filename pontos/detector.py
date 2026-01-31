"""Ship detection using YOLO11s marine vessel model."""

from pathlib import Path
from typing import List, Optional

import torch
from ultralytics import YOLO

from pontos.config import config


class VesselDetector:
    """YOLO11s-based vessel detector for Sentinel-2 imagery."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        device: Optional[str] = None,
        confidence_threshold: float = 0.05,
    ):
        """
        Initialize vessel detector.

        Args:
            model_path: Path to YOLO model weights
            device: Device to run inference on ('0' for GPU, 'cpu' for CPU)
            confidence_threshold: Minimum confidence for detections
        """
        self.model_path = model_path or config.model_path
        self.confidence_threshold = confidence_threshold

        # Smart device selection with fallback
        requested_device = device or config.device
        if requested_device != "cpu" and not torch.cuda.is_available():
            print("GPU requested but not available. Falling back to CPU.")
            self.device = "cpu"
        else:
            self.device = requested_device

        # Load model
        self.model = YOLO(str(self.model_path))

    def detect(
        self,
        image_path: Path,
        save_visualization: bool = False,
        output_dir: Optional[Path] = None,
    ) -> List[dict]:
        """
        Detect vessels in a single image.

        Args:
            image_path: Path to input image
            save_visualization: Whether to save annotated image
            output_dir: Directory to save results

        Returns:
            List of detection dictionaries with bbox coordinates and confidence
        """
        results = self.model(
            str(image_path),
            conf=self.confidence_threshold,
            device=self.device,
            save=save_visualization,
            project=str(output_dir) if output_dir else None,
            verbose=False,
        )

        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])

            detections.append(
                {
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": confidence,
                    "class": self.model.names[class_id],
                    "center": [float((x1 + x2) / 2), float((y1 + y2) / 2)],
                }
            )

        return detections

    def detect_tiled(
        self, image_path: Path, tile_size: int = 320, overlap: float = 0.5
    ) -> List[dict]:
        """
        Detect vessels using sliding window tiling strategy.

        Args:
            image_path: Path to input image
            tile_size: Size of each tile in pixels
            overlap: Overlap ratio between tiles (0.0 to 1.0)

        Returns:
            List of detections with global coordinates
        """
        # TODO: Implement tiled detection with NMS
        raise NotImplementedError("Tiled detection coming in v2.1")

    @property
    def is_gpu_available(self) -> bool:
        """Check if GPU is available and being used."""
        return torch.cuda.is_available() and self.device != "cpu"

    def get_device_name(self) -> str:
        """Get name of device being used for inference."""
        if self.is_gpu_available:
            return torch.cuda.get_device_name(0)
        return "CPU"
