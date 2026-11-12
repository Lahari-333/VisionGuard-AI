"""
VisionGuard AI - Object Detector & Visual HUD Overlay Module
Handles YOLO inference, target filtering, coordinate transformation,
and high-clarity security operations center (SOC) visual annotations.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from config import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    SECURITY_TARGET_CLASSES,
    CLASS_CATEGORIES,
    BOX_COLORS
)
from tracker import TrackedSubject


class VisionDetector:
    """
    Surveillance object detection engine powered by YOLO.
    Performs high-throughput inference and generates clean detection outputs.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        target_classes: Optional[List[str]] = None
    ):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.target_classes = set(target_classes or SECURITY_TARGET_CLASSES)
        self.model = None
        self.device = "cpu"
        self._load_model()

    @property
    def is_loaded(self) -> bool:
        """Returns True if the underlying YOLO model weights are loaded."""
        return self.model is not None

    def _load_model(self):
        """Initializes the YOLO neural network model safely."""
        try:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
        except Exception as e:
            print(f"[VisionGuard Warning] Could not load YOLO model: {e}")
            self.model = None

    def detect(self, frame: np.ndarray, confidence: Optional[float] = None) -> List[Dict]:
        """
        Runs object detection on a BGR image frame.
        Returns a list of detected objects with bounding boxes, labels, and confidence.
        """
        if self.model is None:
            return []

        conf_thresh = confidence if confidence is not None else self.confidence_threshold
        detections = []

        try:
            results = self.model(frame, conf=conf_thresh, verbose=False, device=self.device)
            if not results or len(results) == 0:
                return []

            r = results[0]
            boxes = r.boxes

            for box in boxes:
                cls_id = int(box.cls[0].item())
                cls_name = r.names.get(cls_id, "unknown").lower()
                conf = float(box.conf[0].item())

                # Filter for surveillance-relevant objects if target_classes specified
                if self.target_classes and cls_name not in self.target_classes:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                detections.append({
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "confidence": round(conf, 3),
                    "class_name": cls_name,
                    "class_id": cls_id
                })
        except Exception as e:
            print(f"[VisionGuard Error] Detection inference failed: {e}")

        return detections

    @staticmethod
    def draw_hud(
        frame: np.ndarray,
        tracked_subjects: List[TrackedSubject],
        restricted_zone_norm: List[float],
        threat_level: str = "NORMAL",
        fps: float = 0.0
    ) -> np.ndarray:
        """
        Renders a professional Security Operations Center (SOC) visual overlay:
        - Semi-transparent restricted perimeter with hazard borders
        - Color-coded tracking bounding boxes (Cyan/Amber/Red)
        - Persistent tracking labels and dwell timers
        - Motion trajectory paths
        - Real-time threat status HUD banner
        """
        out = frame.copy()
        h, w = out.shape[:2]

        # 1. Draw Restricted Security Zone
        zx1 = int(restricted_zone_norm[0] * w)
        zy1 = int(restricted_zone_norm[1] * h)
        zx2 = int(restricted_zone_norm[2] * w)
        zy2 = int(restricted_zone_norm[3] * h)

        # Check if zone is currently breached
        any_breach = any(t.is_in_zone and t.class_name == "person" for t in tracked_subjects)
        zone_color = (40, 40, 230) if any_breach else (20, 140, 255)  # Red if breached, Amber/Yellow otherwise

        # Semi-transparent zone overlay
        zone_overlay = out.copy()
        cv2.rectangle(zone_overlay, (zx1, zy1), (zx2, zy2), zone_color, -1)
        alpha = 0.22 if any_breach else 0.12
        cv2.addWeighted(zone_overlay, alpha, out, 1 - alpha, 0, out)

        # Draw zone boundary with dashed/corner accents
        cv2.rectangle(out, (zx1, zy1), (zx2, zy2), zone_color, 2, cv2.LINE_AA)

        # Zone header tag
        zone_label = "RESTRICTED SECURITY ZONE [BREACH DETECTED]" if any_breach else "RESTRICTED SECURITY ZONE"
        label_size, _ = cv2.getTextSize(zone_label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(out, (zx1, zy1 - 20), (zx1 + label_size[0] + 12, zy1), zone_color, -1)
        cv2.putText(out, zone_label, (zx1 + 6, zy1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # 2. Draw Subject Tracks and Trajectories
        for subject in tracked_subjects:
            bx1, by1, bx2, by2 = subject.bbox
            dwell = subject.dwell_time_seconds

            # Determine visual state color
            if subject.is_in_zone and subject.class_name == "person":
                color = BOX_COLORS["critical"]  # Vivid Red
                status_tag = "CRITICAL: ZONE BREACH"
            elif subject.loitering_alert_triggered:
                color = BOX_COLORS["warning"]   # Amber
                status_tag = f"LOITERING ({dwell:.1f}s)"
            else:
                color = BOX_COLORS["normal"]    # Tech Blue/Cyan
                status_tag = f"{dwell:.1f}s"

            # Draw trajectory trail
            if len(subject.trajectory) > 1:
                pts = np.array(subject.trajectory, np.int32).reshape((-1, 1, 2))
                cv2.polylines(out, [pts], False, color, 2, cv2.LINE_AA)

            # Draw target bounding box with corner styling
            cv2.rectangle(out, (bx1, by1), (bx2, by2), color, 2, cv2.LINE_AA)
            
            # Corner accents
            corner_len = min(15, (bx2 - bx1) // 4, (by2 - by1) // 4)
            if corner_len > 3:
                cv2.line(out, (bx1, by1), (bx1 + corner_len, by1), (255, 255, 255), 2)
                cv2.line(out, (bx1, by1), (bx1, by1 + corner_len), (255, 255, 255), 2)
                cv2.line(out, (bx2, by1), (bx2 - corner_len, by1), (255, 255, 255), 2)
                cv2.line(out, (bx2, by1), (bx2, by1 + corner_len), (255, 255, 255), 2)

            # Identification label pill
            tag_text = f"#{subject.track_id} {subject.class_name.upper()} {int(subject.confidence * 100)}% | {status_tag}"
            (tw, th), _ = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            
            label_y = max(by1 - 6, th + 6)
            cv2.rectangle(out, (bx1, label_y - th - 6), (bx1 + tw + 8, label_y + 2), (15, 23, 42), -1)
            cv2.rectangle(out, (bx1, label_y - th - 6), (bx1 + tw + 8, label_y + 2), color, 1)
            cv2.putText(out, tag_text, (bx1 + 4, label_y - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 245, 250), 1, cv2.LINE_AA)

        # 3. Top HUD Status Bar
        hud_bg = out.copy()
        cv2.rectangle(hud_bg, (0, 0), (w, 36), (10, 16, 28), -1)
        cv2.addWeighted(hud_bg, 0.85, out, 0.15, 0, out)
        cv2.line(out, (0, 36), (w, 36), (30, 45, 70), 1)

        # Title & System Status
        title_str = "VISIONGUARD AI | ACTIVE SURVEILLANCE"
        cv2.putText(out, title_str, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

        # Threat Level Badge
        t_color = (50, 50, 235) if threat_level == "CRITICAL" else \
                  (30, 165, 245) if threat_level == "ELEVATED" else (80, 210, 100)
        t_badge = f"THREAT LEVEL: {threat_level}"
        (tb_w, tb_h), _ = cv2.getTextSize(t_badge, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(out, (w - tb_w - 24, 6), (w - 12, 30), t_color, -1)
        cv2.putText(out, t_badge, (w - tb_w - 18, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # Active subject count
        count_str = f"ACTIVE TARGETS: {len(tracked_subjects)}"
        cv2.putText(out, count_str, (w - tb_w - 180, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 200, 220), 1, cv2.LINE_AA)

        return out
