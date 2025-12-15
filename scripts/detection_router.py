#!/usr/bin/env python3
"""
Detection Router - Routes detections based on confidence and model agreement.

Routes each clip into one of four queues:
- AUTO_ACCEPT: High confidence, multi-model agreement -> pseudo-labels for training
- REVIEW: Uncertain cases -> Streamlit for human tagging
- AUTO_REJECT: Low confidence or no detections -> skip training
- NOVEL: Unknown patterns -> CVAT for full annotation

This enables autonomous model improvement with minimal human input.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class RouteDecision(str, Enum):
    """Routing decisions for detections."""
    AUTO_ACCEPT = "auto_accept"  # High confidence, train without review
    REVIEW = "review"           # Uncertain, needs human input
    AUTO_REJECT = "auto_reject"  # Clearly false positive
    NOVEL = "novel"             # New pattern, needs CVAT annotation


@dataclass
class RoutingConfig:
    """Configuration for the routing algorithm."""
    # Auto-accept thresholds
    min_models_agree: int = 2
    min_agreement_confidence: float = 0.85

    # Auto-reject thresholds
    max_reject_confidence: float = 0.20

    # Review thresholds (between reject and accept)
    min_review_confidence: float = 0.25
    max_review_confidence: float = 0.85

    # Novelty detection (placeholder for future embedding-based detection)
    novelty_enabled: bool = False

    # Category weights for scoring
    category_weights: Dict[str, float] = field(default_factory=lambda: {
        "animal": 1.0,
        "person": 1.0,
        "vehicle": 0.5,
    })


@dataclass
class ModelVote:
    """A vote from a single model."""
    model_name: str
    category: str  # "animal", "person", "vehicle"
    confidence: float
    bbox: List[float]
    label: str = ""


@dataclass
class RoutingResult:
    """Result of routing a detection through the system."""
    decision: RouteDecision
    confidence_score: float  # Overall confidence (0-1)
    agreement_score: float   # Model agreement (0-1)
    reason: str             # Human-readable explanation
    votes: List[ModelVote]  # Individual model votes

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision.value,
            "confidence_score": round(self.confidence_score, 3),
            "agreement_score": round(self.agreement_score, 3),
            "reason": self.reason,
            "votes": [
                {
                    "model": v.model_name,
                    "category": v.category,
                    "confidence": round(v.confidence, 3),
                    "label": v.label,
                }
                for v in self.votes
            ],
        }


def extract_votes(model_results: Dict[str, Dict]) -> List[ModelVote]:
    """Extract the best vote from each model's results."""
    votes = []

    for model_name, result in model_results.items():
        hits = result.get("hits", [])
        if not hits:
            continue

        # Find the highest confidence detection
        best_hit = max(hits, key=lambda h: h.get("confidence", 0))

        votes.append(ModelVote(
            model_name=model_name,
            category=best_hit.get("category", "unknown"),
            confidence=best_hit.get("confidence", 0),
            bbox=best_hit.get("bbox", [0, 0, 0, 0]),
            label=best_hit.get("label", ""),
        ))

    return votes


def calculate_agreement(votes: List[ModelVote]) -> Tuple[float, str]:
    """Calculate agreement score and consensus category."""
    if not votes:
        return 0.0, "none"

    if len(votes) == 1:
        return 1.0, votes[0].category

    # Count votes by category
    category_counts: Dict[str, int] = {}
    for vote in votes:
        cat = vote.category
        category_counts[cat] = category_counts.get(cat, 0) + 1

    # Find most common category
    consensus_cat = max(category_counts.keys(), key=lambda c: category_counts[c])
    consensus_count = category_counts[consensus_cat]

    # Agreement = proportion agreeing with consensus
    agreement = consensus_count / len(votes)

    return agreement, consensus_cat


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """Calculate IoU between two [x, y, w, h] boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Convert to [x1, y1, x2, y2]
    b1_x1, b1_y1, b1_x2, b1_y2 = x1, y1, x1 + w1, y1 + h1
    b2_x1, b2_y1, b2_x2, b2_y2 = x2, y2, x2 + w2, y2 + h2

    # Intersection
    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    # Union
    b1_area = w1 * h1
    b2_area = w2 * h2
    union_area = b1_area + b2_area - inter_area

    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def calculate_spatial_agreement(votes: List[ModelVote]) -> float:
    """Calculate how much models agree on location (IoU)."""
    if len(votes) < 2:
        return 1.0

    # Calculate pairwise IoU
    ious = []
    for i, v1 in enumerate(votes):
        for v2 in votes[i + 1:]:
            iou = calculate_iou(v1.bbox, v2.bbox)
            ious.append(iou)

    return sum(ious) / len(ious) if ious else 0.0


def route_detection(
    model_results: Dict[str, Dict],
    config: Optional[RoutingConfig] = None,
) -> RoutingResult:
    """
    Route a detection based on model results.

    Args:
        model_results: Dict mapping model names to their result dicts
                      (as stored in meta.json "models" field)
        config: Routing configuration

    Returns:
        RoutingResult with decision and explanation
    """
    if config is None:
        config = RoutingConfig()

    # Extract votes from each model
    votes = extract_votes(model_results)

    # Case 1: No models detected anything
    if not votes:
        return RoutingResult(
            decision=RouteDecision.AUTO_REJECT,
            confidence_score=0.0,
            agreement_score=0.0,
            reason="No detections from any model",
            votes=[],
        )

    # Calculate metrics
    max_confidence = max(v.confidence for v in votes)
    min_confidence = min(v.confidence for v in votes)
    avg_confidence = sum(v.confidence for v in votes) / len(votes)
    agreement_score, consensus_category = calculate_agreement(votes)
    spatial_agreement = calculate_spatial_agreement(votes)

    # Count agreeing models
    agreeing_models = sum(1 for v in votes if v.category == consensus_category)

    # Case 2: Very low confidence from all models
    if max_confidence < config.max_reject_confidence:
        return RoutingResult(
            decision=RouteDecision.AUTO_REJECT,
            confidence_score=max_confidence,
            agreement_score=agreement_score,
            reason=f"Max confidence {max_confidence:.2f} below reject threshold {config.max_reject_confidence}",
            votes=votes,
        )

    # Case 3: Strong agreement with high confidence -> auto-accept
    if (
        agreeing_models >= config.min_models_agree
        and min_confidence >= config.min_agreement_confidence
        and spatial_agreement >= 0.5  # Boxes should overlap reasonably
    ):
        return RoutingResult(
            decision=RouteDecision.AUTO_ACCEPT,
            confidence_score=avg_confidence,
            agreement_score=agreement_score,
            reason=f"{agreeing_models} models agree on '{consensus_category}' with min_conf={min_confidence:.2f}",
            votes=votes,
        )

    # Case 4: Models disagree on category -> needs review
    if agreement_score < 1.0:
        categories = list(set(v.category for v in votes))
        return RoutingResult(
            decision=RouteDecision.REVIEW,
            confidence_score=max_confidence,
            agreement_score=agreement_score,
            reason=f"Model disagreement: {categories}",
            votes=votes,
        )

    # Case 5: Low spatial agreement -> needs review
    if spatial_agreement < 0.3:
        return RoutingResult(
            decision=RouteDecision.REVIEW,
            confidence_score=avg_confidence,
            agreement_score=agreement_score,
            reason=f"Low spatial agreement (IoU={spatial_agreement:.2f})",
            votes=votes,
        )

    # Case 6: Single model detection with moderate confidence
    if len(votes) == 1:
        if max_confidence >= config.min_review_confidence:
            return RoutingResult(
                decision=RouteDecision.REVIEW,
                confidence_score=max_confidence,
                agreement_score=1.0,
                reason=f"Single model detection ({votes[0].model_name}: {max_confidence:.2f})",
                votes=votes,
            )
        else:
            return RoutingResult(
                decision=RouteDecision.AUTO_REJECT,
                confidence_score=max_confidence,
                agreement_score=1.0,
                reason=f"Single model with low confidence ({max_confidence:.2f})",
                votes=votes,
            )

    # Case 7: Agreement but confidence below auto-accept threshold
    return RoutingResult(
        decision=RouteDecision.REVIEW,
        confidence_score=avg_confidence,
        agreement_score=agreement_score,
        reason=f"Agreement on '{consensus_category}' but confidence below threshold ({min_confidence:.2f} < {config.min_agreement_confidence})",
        votes=votes,
    )


class PseudoLabelStore:
    """Storage for auto-accepted pseudo-labels."""

    def __init__(self, store_path: Path):
        self.store_path = store_path
        self.store_path.mkdir(parents=True, exist_ok=True)
        self.index_path = store_path / "pseudo_label_index.json"
        self._load_index()

    def _load_index(self) -> None:
        """Load the pseudo-label index."""
        if self.index_path.exists():
            self.index = json.loads(self.index_path.read_text())
        else:
            self.index = {
                "clips": {},
                "stats": {"total": 0, "by_category": {}},
            }

    def _save_index(self) -> None:
        """Save the pseudo-label index."""
        self.index_path.write_text(json.dumps(self.index, indent=2))

    def add_clip(
        self,
        clip_path: str,
        routing_result: RoutingResult,
        model_results: Dict[str, Dict],
    ) -> bool:
        """Add a clip to the pseudo-label store."""
        if routing_result.decision != RouteDecision.AUTO_ACCEPT:
            return False

        # Extract consensus category
        if not routing_result.votes:
            return False

        categories = [v.category for v in routing_result.votes]
        consensus = max(set(categories), key=categories.count)

        # Check category caps (prevent drift)
        max_per_category = 500
        current_count = self.index["stats"]["by_category"].get(consensus, 0)
        if current_count >= max_per_category:
            logger.info(f"Pseudo-label cap reached for {consensus} ({current_count})")
            return False

        # Store the clip info
        self.index["clips"][clip_path] = {
            "category": consensus,
            "confidence": routing_result.confidence_score,
            "agreement": routing_result.agreement_score,
            "votes": [{"model": v.model_name, "conf": v.confidence} for v in routing_result.votes],
            "added_at": __import__("datetime").datetime.now().isoformat(),
        }

        # Update stats
        self.index["stats"]["total"] += 1
        self.index["stats"]["by_category"][consensus] = current_count + 1

        self._save_index()
        logger.info(f"Added pseudo-label: {clip_path} ({consensus}, conf={routing_result.confidence_score:.2f})")
        return True

    def get_clips_for_training(self, max_per_category: int = 100) -> Dict[str, List[str]]:
        """Get balanced set of clips for training."""
        by_category: Dict[str, List[str]] = {}

        for clip_path, info in self.index["clips"].items():
            cat = info["category"]
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(clip_path)

        # Sample up to max_per_category from each
        result: Dict[str, List[str]] = {}
        for cat, clips in by_category.items():
            # Sort by confidence (highest first)
            sorted_clips = sorted(
                clips,
                key=lambda c: self.index["clips"][c]["confidence"],
                reverse=True,
            )
            result[cat] = sorted_clips[:max_per_category]

        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the pseudo-label store."""
        return self.index["stats"]


def update_meta_with_routing(
    meta_path: Path,
    routing_result: RoutingResult,
) -> None:
    """Update a meta.json file with routing decision."""
    if not meta_path.exists():
        logger.warning(f"Meta file not found: {meta_path}")
        return

    meta = json.loads(meta_path.read_text())
    meta["routing"] = routing_result.to_dict()
    meta_path.write_text(json.dumps(meta, indent=2))


def batch_route_events(
    events_dir: Path,
    config: Optional[RoutingConfig] = None,
    pseudo_store: Optional[PseudoLabelStore] = None,
) -> Dict[RouteDecision, int]:
    """
    Route all events in a directory.

    Returns counts by decision type.
    """
    counts = {d: 0 for d in RouteDecision}

    if not events_dir.exists():
        return counts

    for day_dir in events_dir.iterdir():
        if not day_dir.is_dir():
            continue

        for event_dir in day_dir.iterdir():
            if not event_dir.is_dir():
                continue

            meta_path = event_dir / "meta.json"
            if not meta_path.exists():
                continue

            try:
                meta = json.loads(meta_path.read_text())

                # Skip if already routed
                if "routing" in meta:
                    decision = RouteDecision(meta["routing"]["decision"])
                    counts[decision] += 1
                    continue

                # Route the detection
                model_results = meta.get("models", {})
                result = route_detection(model_results, config)

                # Update meta
                update_meta_with_routing(meta_path, result)
                counts[result.decision] += 1

                # Add to pseudo-label store if auto-accepted
                if pseudo_store and result.decision == RouteDecision.AUTO_ACCEPT:
                    clip_path = meta.get("segment", str(event_dir))
                    pseudo_store.add_clip(clip_path, result, model_results)

            except Exception as e:
                logger.error(f"Error routing {event_dir}: {e}")

    return counts


if __name__ == "__main__":
    # Test the router
    logging.basicConfig(level=logging.INFO)

    # Example model results
    test_results = {
        "yolov8n_deer": {
            "max_confidence": 0.92,
            "hits": [
                {"category": "animal", "confidence": 0.92, "bbox": [0.3, 0.4, 0.2, 0.3], "label": "deer"},
            ],
        },
        "yolov8s_deer": {
            "max_confidence": 0.88,
            "hits": [
                {"category": "animal", "confidence": 0.88, "bbox": [0.32, 0.38, 0.22, 0.28], "label": "deer"},
            ],
        },
        "rtdetr_deer": {
            "max_confidence": 0.95,
            "hits": [
                {"category": "animal", "confidence": 0.95, "bbox": [0.31, 0.39, 0.21, 0.29], "label": "deer"},
            ],
        },
    }

    result = route_detection(test_results)
    print(f"Decision: {result.decision.value}")
    print(f"Confidence: {result.confidence_score:.2f}")
    print(f"Agreement: {result.agreement_score:.2f}")
    print(f"Reason: {result.reason}")
