"""Pontos ship detection system setup."""
from setuptools import setup, find_packages

setup(
    name="pontos",
    version="0.1.0",
    description="Global naval surveillance using Sentinel-2 and YOLO11s",
    author="Teyk0o",
    packages=find_packages(),
    python_requires=">=3.12",
    install_requires=[
        "ultralytics>=8.3.0",
        "sentinelhub>=3.9.0",
        "numpy",
        "pillow",
        "pyproj",
        "shapely",
        "python-dotenv",
    ],
    entry_points={
        "console_scripts": [
            "pontos=pontos.cli:cli",
        ],
    },
)