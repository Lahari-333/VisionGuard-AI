"""
VisionGuard AI - System Configuration
Original configuration module for VisionGuard AI surveillance system.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Dict

# Base paths
BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Ensure runtime directories exist
ASSETS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Application Identity
APP_NAME = "VISIONGUARD AI"
APP_TAGLINE = "Intelligent Video Surveillance & Real-Time Threat Detection"
APP_VERSION = "1.0.0"

# Target Detection Classes (COCO standard subset relevant to security)
CLASS_CATEGORIES = {
    "person": "HUMAN",
    "car": "VEHICLE",
    "motorcycle": "VEHICLE",
    "bus": "VEHICLE",
    "truck": "VEHICLE",
    "bicycle": "VEHICLE",
    "backpack": "OBJECT",
    "handbag": "OBJECT",
    "suitcase": "OBJECT"
}

SECURITY_TARGET_CLASSES = list(CLASS_CATEGORIES.keys())

# Default Alert Thresholds
DEFAULT_CONFIDENCE_THRESHOLD = 0.40
DEFAULT_IOU_THRESHOLD = 0.45
DEFAULT_LOITERING_THRESHOLD_SEC = 5.0  # seconds visible before loitering alert

# Default Normalized Restricted Zone [x_min, y_min, x_max, y_max] (0.0 to 1.0)
DEFAULT_RESTRICTED_ZONE = [0.25, 0.25, 0.75, 0.75]

# Alert Severity Definitions
SEVERITY_CRITICAL = "Critical"
SEVERITY_WARNING = "Warning"
SEVERITY_NORMAL = "Normal"

# Palette: Professional Security Operations Center (SOC) Dark Theme
PALETTE = {
    "background": "#0B111E",
    "surface": "#131C2E",
    "surface_border": "#1F2D47",
    "primary": "#3B82F6",
    "accent": "#60A5FA",
    "critical": "#EF4444",
    "warning": "#F59E0B",
    "success": "#10B981",
    "text": "#F3F4F6",
    "text_muted": "#9CA3AF"
}

# Bounding Box BGR Colors for OpenCV
BOX_COLORS = {
    "normal": (230, 160, 50),     # Subtle Cyan/Blue (BGR)
    "warning": (20, 165, 245),     # Amber/Orange (BGR)
    "critical": (60, 60, 240)      # Vivid Red (BGR)
}
