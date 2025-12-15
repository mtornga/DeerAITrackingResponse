#!/usr/bin/env python3
"""
Backfill routing decisions on existing events.

This script analyzes all existing events and adds routing decisions
to their meta.json files. This lets us see what the autonomous router
would have decided for clips that were processed before routing was enabled.

Usage:
    python scripts/backfill_routing.py --events-dir /srv/deer-share/runs/live/events
"""

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

# Add repo root to path
script_path = Path(__file__).resolve()
for parent in (script_path.parent, *script_path.parents):
    if (parent / ".env").exists():
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break

from scripts.detection_router import (
    RouteDecision,
    RoutingConfig,
    RoutingResult,
    route_detection,
    PseudoLabelStore,
)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def backfill_events(
    events_dir: Path,
    config: RoutingConfig,
    pseudo_store: PseudoLabelStore = None,
    force: bool = False,
) -> dict:
    """Backfill routing decisions on all events."""
    stats = Counter()
    processed = 0
    skipped = 0

    for day_dir in sorted(events_dir.iterdir()):
        if not day_dir.is_dir():
            continue

        for event_dir in sorted(day_dir.iterdir()):
            if not event_dir.is_dir():
                continue

            meta_path = event_dir / "meta.json"
            if not meta_path.exists():
                continue

            try:
                meta = json.loads(meta_path.read_text())

                # Skip if already has routing (unless force)
                if "routing" in meta and not force:
                    stats[meta["routing"]["decision"]] += 1
                    skipped += 1
                    continue

                # Get model results
                model_results = meta.get("models", {})
                if not model_results:
                    logging.warning(f"No model results in {event_dir}")
                    continue

                # Run routing
                result = route_detection(model_results, config)

                # Update meta
                meta["routing"] = result.to_dict()
                meta_path.write_text(json.dumps(meta, indent=2))

                stats[result.decision.value] += 1
                processed += 1

                # Add to pseudo-label store if auto-accepted
                if pseudo_store and result.decision == RouteDecision.AUTO_ACCEPT:
                    clip_path = meta.get("segment", str(event_dir))
                    pseudo_store.add_clip(clip_path, result, model_results)

                logging.debug(f"  {event_dir.name}: {result.decision.value}")

            except Exception as e:
                logging.error(f"Error processing {event_dir}: {e}")
                stats["error"] += 1

    return {
        "processed": processed,
        "skipped": skipped,
        "by_decision": dict(stats),
    }


def main():
    setup_logging()

    parser = argparse.ArgumentParser(description="Backfill routing decisions on existing events")
    parser.add_argument(
        "--events-dir",
        type=Path,
        default=Path("/srv/deer-share/runs/live/events"),
        help="Events directory to process",
    )
    parser.add_argument(
        "--auto-accept-threshold",
        type=float,
        default=0.85,
        help="Minimum confidence for auto-accept (default: 0.85)",
    )
    parser.add_argument(
        "--min-models",
        type=int,
        default=2,
        help="Minimum models that must agree for auto-accept (default: 2)",
    )
    parser.add_argument(
        "--pseudo-label-dir",
        type=Path,
        default=None,
        help="Directory for pseudo-label storage (enables pseudo-labeling)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-route events that already have routing decisions",
    )
    args = parser.parse_args()

    if not args.events_dir.exists():
        logging.error(f"Events directory not found: {args.events_dir}")
        return 1

    # Set up config
    config = RoutingConfig(
        min_models_agree=args.min_models,
        min_agreement_confidence=args.auto_accept_threshold,
    )

    # Set up pseudo-label store if requested
    pseudo_store = None
    if args.pseudo_label_dir:
        args.pseudo_label_dir.mkdir(parents=True, exist_ok=True)
        pseudo_store = PseudoLabelStore(args.pseudo_label_dir)

    logging.info(f"Backfilling routing decisions in {args.events_dir}")
    logging.info(f"Config: min_models={args.min_models}, auto_accept_threshold={args.auto_accept_threshold}")
    if pseudo_store:
        logging.info(f"Pseudo-labels will be stored in {args.pseudo_label_dir}")

    # Run backfill
    stats = backfill_events(args.events_dir, config, pseudo_store, args.force)

    # Print results
    print("\n" + "=" * 60)
    print("BACKFILL RESULTS")
    print("=" * 60)
    print(f"Events processed: {stats['processed']}")
    print(f"Events skipped (already routed): {stats['skipped']}")
    print(f"\nBy routing decision:")
    for decision, count in sorted(stats["by_decision"].items()):
        print(f"  {decision}: {count}")

    # Calculate percentages
    total = sum(stats["by_decision"].values())
    if total > 0:
        print(f"\nPercentages:")
        for decision, count in sorted(stats["by_decision"].items()):
            pct = 100 * count / total
            print(f"  {decision}: {pct:.1f}%")

        # Show autonomous metrics
        auto_accept = stats["by_decision"].get("auto_accept", 0)
        auto_reject = stats["by_decision"].get("auto_reject", 0)
        review = stats["by_decision"].get("review", 0)

        autonomous_rate = 100 * (auto_accept + auto_reject) / total if total > 0 else 0
        print(f"\n  Autonomous processing rate: {autonomous_rate:.1f}%")
        print(f"  Human review needed: {100 - autonomous_rate:.1f}%")

    if pseudo_store:
        print(f"\nPseudo-label store stats: {pseudo_store.get_stats()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
