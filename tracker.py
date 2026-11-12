"""
VisionGuard AI - Object Tracker Module
Independent multi-object tracking implementation using IoU and Centroid matching.
Designed for reliable identity persistence across video surveillance frames.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import numpy as np


@dataclass
class TrackedSubject:
    """Represents an actively tracked subject in surveillance footage."""
    track_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    centroid: Tuple[int, int]
    first_seen_time: float
    last_seen_time: float
    first_seen_frame: int
    last_seen_frame: int
    disappeared_count: int = 0
    trajectory: List[Tuple[int, int]] = field(default_factory=list)
    
    # State flags for threat detection
    is_in_zone: bool = False
    intrusion_alert_triggered: bool = False
    loitering_alert_triggered: bool = False

    @property
    def dwell_time_seconds(self) -> float:
        """Total active visible dwell duration in seconds."""
        return max(0.0, self.last_seen_time - self.first_seen_time)


def calculate_iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    """Calculates Intersection over Union between two bounding boxes."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])

    inter_width = max(0, xb - xa)
    inter_height = max(0, yb - ya)
    inter_area = inter_width * inter_height

    box_a_area = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    box_b_area = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union_area = float(box_a_area + box_b_area - inter_area)

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


class VisionTracker:
    """
    Lightweight, robust centroid & IoU tracker for real-time video surveillance.
    Assigns persistent IDs without requiring external heavy tracking libraries.
    """

    def __init__(self, max_disappeared: int = 15, iou_threshold: float = 0.25, max_distance: float = 200.0):
        self.next_id: int = 1
        self.tracks: Dict[int, TrackedSubject] = {}
        self.max_disappeared = max_disappeared
        self.iou_threshold = iou_threshold
        self.max_distance = max_distance

    def _compute_centroid(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        x1, y1, x2, y2 = bbox
        return int((x1 + x2) / 2), int((y1 + y2) / 2)

    def update(
        self,
        detections: List[Dict],
        timestamp_sec: float,
        frame_idx: int
    ) -> List[TrackedSubject]:
        """
        Updates the tracker state with new bounding box detections for the current frame.
        
        detections format:
        [
            {"bbox": (x1, y1, x2, y2), "confidence": 0.88, "class_name": "person"},
            ...
        ]
        """
        if len(detections) == 0:
            # Mark all existing tracks as disappeared for this frame
            to_remove = []
            for track_id, track in self.tracks.items():
                track.disappeared_count += 1
                if track.disappeared_count > self.max_disappeared:
                    to_remove.append(track_id)
            for tid in to_remove:
                del self.tracks[tid]
            return [t for t in self.tracks.values() if t.disappeared_count == 0]

        # If no active tracks, initialize all detections
        if len(self.tracks) == 0:
            for det in detections:
                self._register_subject(det, timestamp_sec, frame_idx)
            return list(self.tracks.values())

        track_ids = list(self.tracks.keys())
        track_boxes = [self.tracks[tid].bbox for tid in track_ids]
        track_centroids = [self.tracks[tid].centroid for tid in track_ids]

        det_boxes = [d["bbox"] for d in detections]
        det_centroids = [self._compute_centroid(d["bbox"]) for d in detections]

        num_tracks = len(track_ids)
        num_dets = len(detections)

        # Build cost / distance matrix combining IoU and Euclidean distance
        cost_matrix = np.zeros((num_tracks, num_dets), dtype=np.float32)

        for i in range(num_tracks):
            for j in range(num_dets):
                # Class match preference: same class receives normal match cost
                same_class = (self.tracks[track_ids[i]].class_name == detections[j]["class_name"])
                class_penalty = 0.0 if same_class else 1000.0

                iou_val = calculate_iou(track_boxes[i], det_boxes[j])
                dist = np.hypot(track_centroids[i][0] - det_centroids[j][0],
                                track_centroids[i][1] - det_centroids[j][1])
                
                # Combined metric: prefer high IoU, then low centroid distance
                cost = (1.0 - iou_val) * 100.0 + dist + class_penalty
                cost_matrix[i, j] = cost

        # Greedy assignment based on smallest cost
        assigned_tracks = set()
        assigned_dets = set()

        sorted_indices = np.unravel_index(np.argsort(cost_matrix, axis=None), cost_matrix.shape)

        for r, c in zip(sorted_indices[0], sorted_indices[1]):
            if r in assigned_tracks or c in assigned_dets:
                continue

            # Verification check
            tid = track_ids[r]
            det = detections[c]
            iou_val = calculate_iou(self.tracks[tid].bbox, det["bbox"])
            dist = np.hypot(self.tracks[tid].centroid[0] - det_centroids[c][0],
                            self.tracks[tid].centroid[1] - det_centroids[c][1])

            # Accept match if either IoU meets threshold or centroid distance is close
            if (iou_val >= self.iou_threshold or dist <= self.max_distance) and (self.tracks[tid].class_name == det["class_name"]):
                # Update existing track
                self._update_subject(tid, det, timestamp_sec, frame_idx)
                assigned_tracks.add(r)
                assigned_dets.add(c)

        # Handle unassigned tracks
        for i, tid in enumerate(track_ids):
            if i not in assigned_tracks:
                self.tracks[tid].disappeared_count += 1

        # Handle unassigned detections (new subjects)
        for j, det in enumerate(detections):
            if j not in assigned_dets:
                self._register_subject(det, timestamp_sec, frame_idx)

        # Remove dead tracks
        dead_ids = [tid for tid, track in self.tracks.items() if track.disappeared_count > self.max_disappeared]
        for tid in dead_ids:
            del self.tracks[tid]

        # Return subjects that were visible in the current frame
        return [t for t in self.tracks.values() if t.disappeared_count == 0]

    def _register_subject(self, det: Dict, timestamp_sec: float, frame_idx: int):
        tid = self.next_id
        self.next_id += 1
        centroid = self._compute_centroid(det["bbox"])

        track = TrackedSubject(
            track_id=tid,
            class_name=det["class_name"],
            confidence=det["confidence"],
            bbox=det["bbox"],
            centroid=centroid,
            first_seen_time=timestamp_sec,
            last_seen_time=timestamp_sec,
            first_seen_frame=frame_idx,
            last_seen_frame=frame_idx,
            disappeared_count=0,
            trajectory=[centroid]
        )
        self.tracks[tid] = track

    def _update_subject(self, track_id: int, det: Dict, timestamp_sec: float, frame_idx: int):
        track = self.tracks[track_id]
        centroid = self._compute_centroid(det["bbox"])

        track.bbox = det["bbox"]
        track.centroid = centroid
        track.confidence = det["confidence"]
        track.last_seen_time = timestamp_sec
        track.last_seen_frame = frame_idx
        track.disappeared_count = 0
        track.trajectory.append(centroid)

        # Keep trajectory length bounded to prevent memory growth
        if len(track.trajectory) > 60:
            track.trajectory = track.trajectory[-60:]

    def reset(self):
        """Clears all tracking states."""
        self.tracks.clear()
        self.next_id = 1
