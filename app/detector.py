from typing import List
import logging
import numpy as np
import torch
from app.models import Detection
from app.frame_quality import compute_iou, compute_containment

logger = logging.getLogger(__name__)


def suppress_fragmented_detections(
    detections: List[Detection],
    iou_thresh: float = 0.65,
    containment_thresh: float = 0.80,
) -> List[Detection]:
    if len(detections) <= 1:
        return detections

    sorted_dets = sorted(
        detections,
        key=lambda d: d.confidence * ((d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1])),
        reverse=True,
    )
    kept: List[Detection] = []

    for cand in sorted_dets:
        is_duplicate = False
        for existing in kept:
            iou = compute_iou(cand.bbox, existing.bbox)
            containment = compute_containment(cand.bbox, existing.bbox)
            cx_c = 0.5 * (cand.bbox[0] + cand.bbox[2])
            cx_e = 0.5 * (existing.bbox[0] + existing.bbox[2])
            w_e = max(1, existing.bbox[2] - existing.bbox[0])
            horiz_aligned = abs(cx_c - cx_e) < (0.28 * w_e)

            if iou >= iou_thresh or (containment >= containment_thresh and horiz_aligned):
                is_duplicate = True
                break
        if not is_duplicate:
            kept.append(cand)

    return kept


class PersonDetector:
    def __init__(self, config: dict):
        from ultralytics import YOLO

        det_cfg = config.get("detector", {})
        conf_cfg = config.get("confirmation", {})

        self.model_path = det_cfg.get("model_path", "yolo11n.pt")
        self.imgsz = int(det_cfg.get("imgsz", 640))
        self.conf_threshold = float(det_cfg.get("conf_threshold", 0.45))
        self.candidate_conf = float(det_cfg.get("candidate_conf_threshold", 0.30))
        self.person_class_id = int(det_cfg.get("person_class_id", 0))
        self.iou_thresh = float(conf_cfg.get("nms_iou_threshold", 0.65))
        self.containment_thresh = float(conf_cfg.get("containment_threshold", 0.80))

        req_device = det_cfg.get("device", "auto")
        if req_device == "auto":
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = req_device

        logger.info("Loading YOLO detector (%s) on device=%s", self.model_path, self.device)
        self.model = YOLO(self.model_path)

    def detect_and_track(
        self,
        frame: np.ndarray,
        frame_idx: int,
        timestamp: float,
        tracker_yaml_path: str,
    ) -> List[Detection]:
        results = self.model.track(
            source=frame,
            persist=True,
            tracker=tracker_yaml_path,
            classes=[self.person_class_id],
            conf=self.candidate_conf,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )

        raw_detections: List[Detection] = []
        if not results or len(results) == 0:
            return raw_detections

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return raw_detections

        xyxy = boxes.xyxy.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy().astype(float)
        track_ids = (
            boxes.id.cpu().numpy().astype(int)
            if boxes.id is not None
            else [None] * len(xyxy)
        )

        for bbox, conf, tid in zip(xyxy, confs, track_ids):
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            det = Detection(
                bbox=(x1, y1, x2, y2),
                confidence=float(conf),
                frame_idx=frame_idx,
                timestamp=timestamp,
                track_id=int(tid) if tid is not None else None,
            )
            if conf < self.conf_threshold:
                det.uncertainty_flags.append("low_detection_confidence")
            raw_detections.append(det)

        return suppress_fragmented_detections(
            raw_detections,
            iou_thresh=self.iou_thresh,
            containment_thresh=self.containment_thresh,
        )
