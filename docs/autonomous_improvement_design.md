# Autonomous Model Improvement System Design

**Created**: 2025-12-15
**Status**: Design Document
**Goal**: Maximize model improvement velocity while minimizing human input requirements

## Problem Statement

From NewNotes.txt:
> "The agents should work as autonomously as possible but leverage mtornga only as much as necessary."

Current system requires human review for every detected clip. With 100+ clips/day during active wildlife periods, this doesn't scale. We need a system that:

1. **Auto-accepts** clear detections (high confidence, multi-model agreement)
2. **Auto-rejects** clear false positives (learned patterns)
3. **Routes to human** only genuinely uncertain or novel cases
4. **Self-improves** using high-confidence predictions as training data

## Architecture Overview

```
                    ┌─────────────────────────────────────────┐
                    │           Live Detection                │
                    │  (YOLOv8n + YOLOv8s + RT-DETR)         │
                    └─────────────────┬───────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────────┐
                    │         Confidence Router               │
                    │  - Multi-model voting                   │
                    │  - Uncertainty estimation               │
                    │  - Novelty detection                    │
                    └───────┬─────────────┬───────────────────┘
                            │             │
              ┌─────────────┼─────────────┼─────────────┐
              ▼             ▼             ▼             ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
        │AUTO-ACCEPT│  │  REVIEW  │  │AUTO-REJECT│  │  NOVEL   │
        │  Queue   │  │  Queue   │  │  Queue   │  │  Queue   │
        │ (>95%)   │  │ (50-95%) │  │  (<20%)  │  │ (unknown)│
        └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘
             │             │             │             │
             ▼             ▼             │             ▼
     ┌───────────────┐    │             │      ┌───────────────┐
     │Pseudo-Label DB│    │             │      │ CVAT Inbox    │
     │(auto training)│    │             │      │(full labeling)│
     └───────┬───────┘    │             │      └───────┬───────┘
             │             │             │              │
             │      ┌──────▼──────┐      │              │
             │      │  Streamlit  │      │              │
             │      │Morning Queue│      │              │
             │      │(quick tags) │      │              │
             │      └──────┬──────┘      │              │
             │             │             │              │
             └─────────────┴─────────────┴──────────────┘
                           │
                           ▼
                    ┌─────────────────────────────────────────┐
                    │         Feedback Training               │
                    │  - Verified human labels                │
                    │  - High-confidence pseudo-labels        │
                    │  - Hard negative mining                 │
                    └─────────────────────────────────────────┘
```

## Component Details

### 1. Confidence Router

The router examines each detection and routes it to the appropriate queue.

**Multi-Model Voting Logic**:
```python
def route_detection(detections: Dict[str, List[Detection]]) -> str:
    """Route based on multi-model agreement and confidence."""

    # Collect votes from all models
    votes = []
    for model, dets in detections.items():
        if dets:
            best = max(dets, key=lambda d: d.confidence)
            votes.append({
                'model': model,
                'class': best.class_name,
                'confidence': best.confidence
            })

    # Case 1: All models agree on class with high confidence
    if len(votes) >= 2:
        classes = set(v['class'] for v in votes)
        min_conf = min(v['confidence'] for v in votes)

        if len(classes) == 1 and min_conf > 0.85:
            return 'AUTO_ACCEPT'  # Strong consensus

    # Case 2: No detections from any model
    if len(votes) == 0:
        return 'AUTO_REJECT'  # True negative

    # Case 3: Models disagree or low confidence
    confidences = [v['confidence'] for v in votes]
    if max(confidences) < 0.3:
        return 'AUTO_REJECT'  # Likely noise

    if max(confidences) < 0.7 or len(set(v['class'] for v in votes)) > 1:
        return 'REVIEW'  # Human needed

    # Case 4: Single model detection, moderate confidence
    return 'REVIEW'
```

**Uncertainty Estimation**:
- **Confidence spread**: Large gap between models = uncertainty
- **Box overlap**: Low IoU between model boxes = uncertainty
- **Temporal consistency**: Detection that appears/disappears quickly = suspicious

### 2. Auto-Accept Queue (Pseudo-Labeling)

Clips where all models agree with high confidence become "pseudo-labels":

```python
class PseudoLabelManager:
    """Manages auto-accepted detections for training."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.min_agreement = 2  # At least 2 models must agree
        self.min_confidence = 0.90
        self.max_pseudo_per_class = 500  # Cap to prevent drift

    def add_if_qualified(self, clip_path: str, detections: Dict):
        """Add to pseudo-label pool if quality threshold met."""

        # Extract consensus detections
        consensus = self.find_consensus(detections)
        if not consensus:
            return False

        # Check class balance
        class_counts = self.get_class_counts()
        for det in consensus:
            if class_counts.get(det.class_name, 0) >= self.max_pseudo_per_class:
                continue  # Skip over-represented classes

            self.store_pseudo_label(clip_path, det)

        return True

    def export_for_training(self, output_dir: Path):
        """Export pseudo-labels in YOLO format for training."""
        # Sample balanced subset
        # Convert to YOLO annotation format
        # Mix with human-verified data (70% human, 30% pseudo)
```

**Safety Mechanisms**:
- Cap pseudo-labels per class to prevent drift
- Periodic human audit (sample 5% of auto-accepts weekly)
- Automatic rollback if model performance drops

### 3. Review Queue (Streamlit - Simplified)

Only uncertain clips reach Streamlit. Interface optimized for speed:

```
┌─────────────────────────────────────────────────────────┐
│  Morning Review: 8 clips need attention (est. 2 min)   │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  [VIDEO PLAYER - 10 sec clip]                          │
│                                                         │
│  Models say:  YOLOv8n: deer (0.72)                     │
│               YOLOv8s: person (0.68)  ← DISAGREEMENT   │
│               RT-DETR: deer (0.81)                     │
│                                                         │
│  Quick Tags:  [DEER] [PERSON] [FALSE +] [SKIP]         │
│                                                         │
│  Keyboard:    D=Deer  P=Person  F=False+  →=Skip       │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**Human Input Maximization**:
- Show WHY clip was flagged (model disagreement, low confidence, etc.)
- One-click tagging with keyboard shortcuts
- Batch similar-looking clips for efficient review
- Estimated time to completion

### 4. Novel/CVAT Queue

Truly novel situations (new lighting, new animal species, unusual behavior) go to CVAT for full bounding box annotation:

**Novelty Detection**:
```python
def is_novel(frame: np.ndarray, detection: Detection) -> bool:
    """Detect if this is a genuinely new situation."""

    # Extract features from detection crop
    crop = extract_crop(frame, detection.box)
    embedding = feature_extractor(crop)

    # Compare to known embeddings database
    distances = cosine_distance(embedding, known_embeddings)
    min_distance = distances.min()

    # Novel if far from all known examples
    return min_distance > NOVELTY_THRESHOLD
```

**CVAT Integration**:
```python
class CVATAutoAnnotator:
    """Deploy model to CVAT for assisted labeling."""

    def __init__(self, cvat_url: str, model_path: str):
        self.api = CVAT(cvat_url)
        self.model = load_model(model_path)

    def create_task_with_predictions(self, images: List[Path], task_name: str):
        """Create CVAT task with model pre-annotations."""

        # Create task
        task = self.api.create_task(task_name)

        # Upload images
        self.api.upload_images(task.id, images)

        # Generate predictions
        predictions = []
        for img_path in images:
            img = cv2.imread(str(img_path))
            dets = self.model(img)
            predictions.extend(self.format_for_cvat(img_path, dets))

        # Upload as pre-annotations
        self.api.upload_annotations(task.id, predictions)

        return task.url  # mtornga can review/correct
```

### 5. Feedback Training Loop

The training loop automatically incorporates new data:

```python
class AutoTrainingLoop:
    """Autonomous training without human intervention."""

    def __init__(self, config: TrainConfig):
        self.config = config
        self.performance_history = []

    def should_retrain(self) -> bool:
        """Decide if retraining is warranted."""

        # Check if enough new data accumulated
        new_labels = count_new_labels_since_last_train()
        if new_labels < 50:
            return False

        # Check if performance has degraded
        recent_fp_rate = get_recent_false_positive_rate()
        if recent_fp_rate > 0.10:  # More than 10% FP
            return True

        # Weekly scheduled retrain
        if days_since_last_train() >= 7:
            return True

        return False

    def run_training_cycle(self):
        """Execute full training cycle."""

        # 1. Merge datasets
        dataset = merge_datasets(
            human_verified=self.config.human_labels_dir,
            pseudo_labels=self.config.pseudo_labels_dir,
            hard_negatives=self.config.hard_neg_dir,
            pseudo_ratio=0.3  # Cap pseudo-labels at 30%
        )

        # 2. Train all models
        results = {}
        for model_type in ['yolov8n', 'yolov8s', 'rtdetr-l']:
            try:
                metrics = train_model(model_type, dataset)
                results[model_type] = metrics
            except Exception as e:
                log_error(f"Training {model_type} failed: {e}")

        # 3. Evaluate on held-out test set
        for model_type, metrics in results.items():
            test_metrics = evaluate_on_test_set(model_type)
            results[model_type]['test'] = test_metrics

        # 4. Promote only if improved
        for model_type, metrics in results.items():
            if self.is_improvement(model_type, metrics):
                promote_model(model_type)
                log_info(f"Promoted {model_type}: {metrics}")
            else:
                log_warning(f"Rejected {model_type}: no improvement")

        # 5. Archive experiment for reproducibility
        archive_experiment(dataset, results)
```

## Implementation Phases

### Phase 1: Confidence Router (This Week)
- Add confidence routing to `live_detector_multimodel.py`
- Create auto-accept and auto-reject queues
- Log routing decisions for analysis

### Phase 2: Pseudo-Label Pipeline (Next Week)
- Build pseudo-label storage and export
- Add to training pipeline with configurable ratio
- Implement class balance caps

### Phase 3: CVAT Integration (Week 3)
- Deploy trained model to CVAT as auto-annotator
- Create workflow for novel sample annotation
- Build export pipeline from CVAT to training

### Phase 4: Full Autonomy (Week 4)
- Automated retraining triggers
- Self-monitoring and rollback
- Performance dashboards

## Metrics to Track

### Autonomous Efficiency
- **Auto-accept rate**: % of clips not requiring human review
- **Review time**: Minutes spent in Streamlit per day
- **Pseudo-label accuracy**: Spot-check accuracy of auto-accepts

### Model Quality
- **False positive rate**: FP per hour of video
- **Recall by class**: Deer, person, unknown_animal
- **Novel detection rate**: New situations caught vs missed

### Training Velocity
- **Days between retrains**: Target: 7 days or less
- **New samples per cycle**: Human + pseudo combined
- **Model version history**: Track improvements over time

## Risk Mitigation

### Model Drift
- **Problem**: Pseudo-labels could reinforce errors
- **Solution**: Cap pseudo-label ratio, weekly human audits

### Class Imbalance
- **Problem**: Too many deer, not enough person/skunk
- **Solution**: Per-class caps on pseudo-labels

### Catastrophic Forgetting
- **Problem**: New training forgets old edge cases
- **Solution**: Maintain frozen eval set, rollback on degradation

### Hardware Failures
- **Problem**: Mount disconnects, GPU OOM
- **Solution**: Already addressed with LaunchAgent mount, batch size tuning

## Daily Workflow (Target State)

**Morning (5 minutes)**:
1. Open Streamlit
2. See "8 clips need review" (down from 100+)
3. Quick-tag with keyboard shortcuts
4. Done

**Background (Autonomous)**:
- Router categorizes all new detections
- High-confidence clips auto-added to training pool
- Weekly retrain runs overnight
- Performance monitored, alerts on degradation

**Weekly (15 minutes)**:
- Audit sample of auto-accepts
- Review CVAT inbox for novel situations
- Check performance dashboard

## Next Steps

1. Implement confidence router in `scripts/route_detections.py`
2. Add pseudo-label storage to `scripts/feedback_cycle.py`
3. Test CVAT API for auto-annotation
4. Build monitoring dashboard for autonomous metrics
