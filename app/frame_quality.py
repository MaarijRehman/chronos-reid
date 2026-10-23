from typing import List, Tuple
import cv2
import numpy as np
from app.models import Detection


def compute_iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    inter_w = max(0, x_b - x_a)
    inter_h = max(0, y_b - y_a)
    inter_area = float(inter_w * inter_h)
    if inter_area <= 0.0:
        return 0.0

    area_a = float(max(1, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])))
    area_b = float(max(1, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])))
    return inter_area / (area_a + area_b - inter_area)


def compute_containment(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    inter_area = float(max(0, x_b - x_a) * max(0, y_b - y_a))
    if inter_area <= 0.0:
        return 0.0

    area_a = float(max(1, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])))
    area_b = float(max(1, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])))
    return inter_area / min(area_a, area_b)


class FrameQualityAssessor:
    def __init__(self, config: dict):
        q_cfg = config.get("quality", {})
        c_cfg = config.get("confirmation", {})
        d_cfg = config.get("detector", {})

        self.blur_ref = float(q_cfg.get("blur_laplacian_ref", 120.0))
        self.w_conf = float(q_cfg.get("weight_conf", 0.30))
        self.w_sharp = float(q_cfg.get("weight_sharpness", 0.25))
        self.w_size = float(q_cfg.get("weight_size", 0.20))
        self.w_bound = float(q_cfg.get("weight_boundary", 0.15))
        self.w_occ = float(q_cfg.get("weight_occlusion", 0.10))

        self.min_w = int(c_cfg.get("min_crop_width", 32))
        self.min_h = int(c_cfg.get("min_crop_height", 64))
        self.edge_margin = int(d_cfg.get("edge_margin_px", 12))

    def compute_sharpness(self, img: np.ndarray) -> float:
        if img is None or img.size == 0:
            return 0.0
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        return float(np.clip(lap_var / self.blur_ref, 0.0, 1.0))

    def assess_detections(self, frame: np.ndarray, detections: List[Detection]) -> float:
        h, w = frame.shape[:2]
        frame_area = float(max(1, w * h))
        frame_sharpness = self.compute_sharpness(frame)

        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det.bbox
            x1_c, y1_c = max(0, x1), max(0, y1)
            x2_c, y2_c = min(w, x2), min(h, y2)

            bw = x2_c - x1_c
            bh = y2_c - y1_c
            det.crop = frame[y1_c:y2_c, x1_c:x2_c].copy() if (bw > 0 and bh > 0) else None

            det.touches_boundary = (
                x1_c <= self.edge_margin
                or y1_c <= self.edge_margin
                or x2_c >= (w - self.edge_margin)
                or y2_c >= (h - self.edge_margin)
            )
            if det.touches_boundary:
                det.uncertainty_flags.append("boundary_truncated")

            max_overlap = 0.0
            for j, other in enumerate(detections):
                if i == j:
                    continue
                max_overlap = max(max_overlap, compute_iou(det.bbox, other.bbox))
            det.is_occluded = max_overlap > 0.25
            if det.is_occluded:
                det.uncertainty_flags.append("partially_occluded")

            if bw < self.min_w or bh < self.min_h or det.crop is None:
                det.sharpness_score = 0.0
                det.quality_score = 0.0
                det.uncertainty_flags.append("below_min_dimensions")
                continue

            det.sharpness_score = self.compute_sharpness(det.crop)
            rel_area = (bw * bh) / frame_area
            size_score = float(np.clip(rel_area / 0.06, 0.15, 1.0))
            boundary_score = 0.55 if det.touches_boundary else 1.0
            occlusion_score = float(np.clip(1.0 - max_overlap, 0.2, 1.0))

            composite = (
                self.w_conf * det.confidence
                + self.w_sharp * det.sharpness_score
                + self.w_size * size_score
                + self.w_bound * boundary_score
                + self.w_occ * occlusion_score
            )
            det.quality_score = float(np.clip(composite, 0.0, 1.0))

        if not detections:
            return frame_sharpness
        avg_det_q = float(np.mean([d.quality_score for d in detections]))
        return float(0.5 * frame_sharpness + 0.5 * avg_det_q)
