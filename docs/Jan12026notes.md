# January 1, 2026 - Session Notes

## Summary
Continued FP reduction work on the DeerVision pipeline. Added multiple filters to combat false positives from various sources.

## FP Patterns Identified & Fixed

### 1. Daytime Grass Texture FPs (Dec 31)
- **Pattern**: Static detections at x=0.19, y=0.63 with tiny bbox (3.8%×9.6%)
- **Fix**: Static detection filter (variance < 0.02) + bbox size filter (min 4%×8%)

### 2. Nighttime Corner FPs (overnight Jan 1)
- **Pattern**: RT-DETR detecting camera overlay/timestamp at [0.0, 0.0] as "deer"
- **Fix**: Edge exclusion filter (reject detections within 5% of frame edge)

### 3. IR Bug Flash FPs
- **Pattern**: Bugs lit by IR create 3-5 detections on 1-3 frames (single-frame bursts)
- **Fix**: Minimum frame span filter - require detections to span 45+ frames (~3 seconds)

## Current Filter Stack
```
1. Confidence threshold: 0.40
2. Temporal filter: 6+ detections spanning 45+ frames
3. Bbox size filter: min 4% × 8%
4. Static detection filter: position variance > 0.02
5. Edge exclusion filter: 5% margin from frame edges
```

## Pipeline Configuration (.env on server)
```bash
DETECTOR_CONFIDENCE="0.40"
DETECTOR_MIN_CLUSTER_DETECTIONS="6"
DETECTOR_MIN_FRAME_SPAN="45"
LIGHTING_MODE="day"  # Currently forced; should use "auto" normally
```

## Known Issues

### Dusk Detection Problem
- At ~4:30 PM Central (near sunset), auto-lighting picks "night" even though camera may still be in color mode
- Day model detects person at only 25-30% confidence (below threshold)
- YOLO day model detects nothing; only RT-DETR sees faint detections
- **TODO**: May need to retrain day models with more dusk/low-light examples, or lower threshold during transition hours

### Segment Backlog
- Pipeline can accumulate massive backlog (6000+ segments) if detector falls behind
- Solution: Periodically clear old segments with `find ... -mmin +N -delete`

## Files Changed
- `scripts/live_detector_multimodel.py` - Added edge exclusion filter, min_frame_span parameter
- `scripts/run_pipeline.sh` - Added new filter parameters
- `CLAUDE.md` - Added location info (St. Louis, MO - Central Time UTC-6)

## Commits
- `feat: add FP reduction filters (bbox size, static detection, model agreement)`
- `feat: add edge exclusion filter and increase temporal threshold to 4`
- `feat: add min_frame_span filter to catch bug flashes`

## Next Steps
1. Test filters overnight to verify bug FPs are eliminated
2. Investigate dusk detection - may need threshold adjustment for transition hours
3. Consider training v8 day models with more varied lighting conditions
4. Review any new FPs that slip through to identify additional patterns
