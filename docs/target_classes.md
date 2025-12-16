# Target Detection Classes

**Created**: 2025-12-16
**Status**: Planning
**Goal**: Define all classes the wildlife detection system should identify

## Overview

The system needs to distinguish between:
1. **Deterrent targets** - Wildlife to redirect (deer, etc.)
2. **Protected entities** - Never target (people, UGV)
3. **Monitored wildlife** - Track but may not deter (predators, small animals)

## Class Definitions

### Tier 1: Primary Targets (Train First)

| Class | Priority | Data Status | Notes |
|-------|----------|-------------|-------|
| `deer` | **Critical** | Have training data | Primary deterrent target. Whitetail deer. |
| `person` | **Critical** | Use COCO + walk-by clips | Safety critical - UGV must never approach humans |

### Tier 2: Common Wildlife (Train Next)

| Class | Priority | Data Status | Notes |
|-------|----------|-------------|-------|
| `raccoon` | High | Need to collect | Nocturnal, common visitor |
| `opossum` | High | Need to collect | Nocturnal, slow-moving |
| `skunk` | High | Need to collect | Nocturnal, distinctive markings |
| `fox` | Medium | Need to collect | Less common, crepuscular |
| `coyote` | Medium | Need to collect | Safety concern, larger predator |
| `groundhog` | Medium | Need to collect | Daytime, distinctive shape |

### Tier 3: Small/Difficult (Consider Carefully)

| Class | Priority | Data Status | Notes |
|-------|----------|-------------|-------|
| `squirrel` | Low | Abundant in COCO | **High noise potential** - very common, may trigger constantly. Consider: ignore, or train but don't alert. |
| `weasel` | Low | Rare, need to collect | Small, fast, hard to detect |

### Tier 4: Reptiles & Birds

| Class | Priority | Data Status | Notes |
|-------|----------|-------------|-------|
| `snake` | Low | Need to collect | Ground-level, elongated shape |
| `skink` | Low | Need to collect | Very small lizard, may be below detection threshold |
| `owl` | Low | Need to collect | Nocturnal, often stationary |
| `hawk` | Low | Need to collect | Daytime, aerial or perched |

### Tier 5: System Entities (Future)

| Class | Priority | Data Status | Notes |
|-------|----------|-------------|-------|
| `ugv` | Future | **Not available yet** | CuteBot robot. Need to deploy first, then capture training data. Important for: avoiding self-detection, coordination. |

## Training Strategy

### Phase 1: Core Detection (Now)
- **Classes**: `deer`, `person`
- **Models**: YOLOv8n, RT-DETR
- **Data sources**:
  - Deer: Existing labeled clips from outdoor camera
  - Person: COCO subset + walk-by test clips

### Phase 2: Nocturnal Wildlife
- **Classes**: Add `raccoon`, `opossum`, `skunk`
- **Challenge**: IR/night vision imagery differs from daytime
- **Data sources**: Collect from live system, label in CVAT

### Phase 3: Predators & Ground Mammals
- **Classes**: Add `fox`, `coyote`, `groundhog`
- **Data sources**: Live collection + possibly iNaturalist/wildlife datasets

### Phase 4: Edge Cases
- **Classes**: Add birds, reptiles, small mammals as data becomes available
- **Squirrel decision**: Test with squirrel class, measure false positive rate, decide whether to keep

### Phase 5: UGV Self-Recognition
- **Classes**: Add `ugv`
- **Prerequisite**: Deploy CuteBot outdoors, capture footage from multiple angles

## Class Hierarchy for Routing

```
Detection
├── PROTECTED (never deter)
│   ├── person
│   └── ugv
├── DETER (redirect with UGV)
│   └── deer
├── MONITOR (log but don't deter)
│   ├── predators: fox, coyote
│   ├── nocturnal: raccoon, opossum, skunk
│   ├── small_mammals: squirrel, groundhog, weasel
│   ├── birds: owl, hawk
│   └── reptiles: snake, skink
└── IGNORE (don't log)
    └── (TBD based on noise analysis)
```

## YOLO Class Configuration

For training, the `data.yaml` will look like:

```yaml
# Phase 1 config
names:
  0: deer
  1: person

# Future full config
names:
  0: deer
  1: person
  2: raccoon
  3: opossum
  4: skunk
  5: fox
  6: coyote
  7: groundhog
  8: squirrel
  9: weasel
  10: snake
  11: skink
  12: owl
  13: hawk
  14: ugv
```

## Data Collection Notes

### Sources
- **Live system**: Primary source for all classes
- **COCO**: Person, some animals (but wrong context - indoor/urban)
- **iNaturalist**: Wildlife images (but may lack bounding boxes)
- **Caltech Camera Traps**: Trail camera data (similar domain)
- **LILA BC**: Large wildlife camera trap datasets

### Labeling Workflow
1. Collect clips from live system
2. Auto-label with MegaDetector (animal/person/vehicle)
3. Refine labels in CVAT with specific class
4. Export to YOLO format
5. Merge with existing training data

## Open Questions

1. **Squirrel threshold**: At what false-positive rate do we disable squirrel detection?
2. **Night vs day models**: Train separate models or unified with augmentation?
3. **Minimum viable size**: What's the smallest animal we can reliably detect at camera distance?
4. **Bird in flight**: Do we need to detect flying birds or only perched?
