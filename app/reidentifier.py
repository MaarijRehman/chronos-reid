import logging
from typing import Optional
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T

logger = logging.getLogger(__name__)


class AppearanceEmbedder:
    def __init__(self, config: dict):
        reid_cfg = config.get("reid", {})
        det_cfg = config.get("detector", {})
        self.backbone_name = reid_cfg.get("backbone", "osnet_x1_0")
        self.weights_path = reid_cfg.get("weights_path", "")

        req_device = det_cfg.get("device", "auto")
        if req_device == "auto":
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(req_device)

        self.extractor = None
        self.fallback_model: Optional[nn.Module] = None
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self._init_model()

    def _init_model(self) -> None:
        if self.backbone_name.startswith("osnet"):
            try:
                from torchreid.utils import FeatureExtractor
                self.extractor = FeatureExtractor(
                    model_name=self.backbone_name,
                    model_path=self.weights_path if self.weights_path else "",
                    device=str(self.device),
                    verbose=False,
                )
                logger.info("Initialized torchreid FeatureExtractor (%s) on %s", self.backbone_name, self.device)
                return
            except Exception as exc:
                logger.warning(
                    "torchreid (%s) unavailable (%s); using Torchvision MobileNetV3 ReID backbone.",
                    self.backbone_name,
                    exc,
                )

        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
        base = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        self.fallback_model = base.features.to(self.device).eval()
        logger.info("Initialized Torchvision MobileNetV3-Small ReID backbone on %s", self.device)

    @staticmethod
    def l2_normalize(vec: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vec))
        if norm < 1e-8:
            return vec
        return (vec / norm).astype(np.float32)

    @staticmethod
    def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        return float(np.dot(vec_a, vec_b))

    @torch.no_grad()
    def extract(self, crop_bgr: np.ndarray) -> Optional[np.ndarray]:
        if crop_bgr is None or crop_bgr.size == 0:
            return None

        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

        if self.extractor is not None:
            feats = self.extractor([crop_rgb])
            vec = feats[0].cpu().numpy().flatten().astype(np.float32)
            return self.l2_normalize(vec)

        tensor = self.transform(crop_rgb).unsqueeze(0).to(self.device)
        feat_map = self.fallback_model(tensor)
        h = feat_map.shape[2]
        top_half = feat_map[:, :, : max(1, h // 2), :].mean(dim=(2, 3))
        bot_half = feat_map[:, :, max(1, h // 2) :, :].mean(dim=(2, 3))
        global_pool = feat_map.mean(dim=(2, 3))
        combined = torch.cat([global_pool, top_half, bot_half], dim=1)
        vec = combined[0].cpu().numpy().flatten().astype(np.float32)
        return self.l2_normalize(vec)
