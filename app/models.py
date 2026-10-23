from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple
import numpy as np


class IdentityState(str, Enum):
    CANDIDATE = "CANDIDATE"
    ACTIVE = "ACTIVE"
    TEMPORARILY_MISSING = "TEMPORARILY_MISSING"
    ABSENT = "ABSENT"
    RETURN_CANDIDATE = "RETURN_CANDIDATE"


class EventType(str, Enum):
    NEW_PERSON = "new_person"
    RETURN = "return"


@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]
    confidence: float
    frame_idx: int
    timestamp: float
    track_id: Optional[int] = None
    crop: Optional[np.ndarray] = None
    quality_score: float = 0.0
    sharpness_score: float = 0.0
    touches_boundary: bool = False
    is_occluded: bool = False
    embedding: Optional[np.ndarray] = None
    global_id: Optional[str] = None
    match_score: float = 0.0
    uncertainty_flags: List[str] = field(default_factory=list)


@dataclass
class CandidateRecord:
    candidate_id: str
    first_seen_frame: int
    first_seen_ts: float
    last_seen_frame: int
    last_seen_ts: float
    track_ids: Set[int] = field(default_factory=set)
    observation_frames: List[int] = field(default_factory=list)
    valid_observation_frames: List[int] = field(default_factory=list)
    embeddings: List[np.ndarray] = field(default_factory=list)
    confidences: List[float] = field(default_factory=list)
    qualities: List[float] = field(default_factory=list)
    best_crop: Optional[np.ndarray] = None
    best_quality: float = 0.0
    best_bbox: Optional[Tuple[int, int, int, int]] = None
    provisional_target_id: Optional[str] = None
    uncertainty_flags: Set[str] = field(default_factory=set)


@dataclass
class IdentityRecord:
    global_id: str
    state: IdentityState
    first_seen_frame: int
    first_seen_ts: float
    last_seen_frame: int
    last_seen_ts: float
    last_confirmed_visible_ts: float
    associated_track_ids: Set[int] = field(default_factory=set)
    embeddings: List[np.ndarray] = field(default_factory=list)
    embedding_qualities: List[float] = field(default_factory=list)
    crops: List[Tuple[float, np.ndarray]] = field(default_factory=list)
    confidence_history: List[float] = field(default_factory=list)
    quality_history: List[float] = field(default_factory=list)
    match_score_history: List[float] = field(default_factory=list)
    num_observations: int = 0
    appearance_episodes: int = 1
    event_history: List[Dict] = field(default_factory=list)
    disappearance_history: List[Tuple[float, Optional[float]]] = field(default_factory=list)
    uncertainty_flags: Set[str] = field(default_factory=set)
    return_obs_frames: List[int] = field(default_factory=list)
    captured_episodes: Set[Tuple[str, int]] = field(default_factory=set)


@dataclass
class PendingEvent:
    event_key: str
    event_type: EventType
    global_id: str
    episode_id: int
    trigger_frame: int
    trigger_ts: float
    confirmation_info: Dict
    match_diagnostics: Dict


@dataclass
class FramePacket:
    frame_idx: int
    timestamp: float
    frame: np.ndarray
    detections: List[Detection]
    active_global_ids: Set[str]
    camera_mode_state: str
    camera_motion_px: float
    frame_quality_score: float


@dataclass
class FinalizedScreenshotEvent:
    event_id: str
    event_keys: List[str]
    screenshot_filename: str
    frame_number: int
    timestamp: float
    event_types: List[str]
    primary_event_type: str
    triggering_global_ids: List[str]
    all_visible_global_ids: List[str]
    tracker_ids: List[int]
    bboxes: Dict[str, List[int]]
    confidences: Dict[str, float]
    reid_diagnostics: Dict[str, Dict]
    confirmation_info: Dict[str, Dict]
    frame_quality_score: float
    uncertainty_flags: List[str]
    camera_motion_state: str
