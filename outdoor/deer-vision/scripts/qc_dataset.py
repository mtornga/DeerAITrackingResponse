"""Quality checks for YOLO-format datasets."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from utils import IMAGE_EXTENSIONS  # noqa: E402


@dataclass
class LabelIssue:
    file: str
    line: int
    message: str


def validate_label_file(path: Path, allowed_classes: List[int]) -> List[LabelIssue]:
    issues: List[LabelIssue] = []
    with path.open() as handle:
        for idx, raw in enumerate(handle, start=1):
            raw = raw.strip()
            if not raw:
                continue
            parts = raw.split()
            if len(parts) < 5:
                issues.append(LabelIssue(str(path), idx, "Line must have at least 5 values"))
                continue
            try:
                cls_id = int(parts[0])
                values = list(map(float, parts[1:5]))
            except ValueError:
                issues.append(LabelIssue(str(path), idx, "Non-numeric value detected"))
                continue
            if allowed_classes and cls_id not in allowed_classes:
                issues.append(LabelIssue(str(path), idx, f"Class {cls_id} not in allowed list"))
            if any(v < 0 or v > 1 for v in values):
                issues.append(LabelIssue(str(path), idx, "Coordinates must be normalized between 0 and 1"))
            if values[2] <= 0 or values[3] <= 0:
                issues.append(LabelIssue(str(path), idx, "Width/height must be positive"))
    return issues


def verify_image_pairs(labels_root: Path, images_root: Path | None) -> List[LabelIssue]:
    issues: List[LabelIssue] = []
    if images_root is None:
        return issues
    for label in labels_root.rglob("*.txt"):
        image_candidate = images_root / label.relative_to(labels_root)
        image_candidate = image_candidate.with_suffix(".jpg")
        if not image_candidate.exists():
            # Try other extensions
            found = False
            for ext in IMAGE_EXTENSIONS:
                alt = image_candidate.with_suffix(ext)
                if alt.exists():
                    found = True
                    break
            if not found:
                issues.append(LabelIssue(str(label), 0, "Missing paired image"))
    return issues


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run QC checks on YOLO labels.")
    parser.add_argument("--labels", type=Path, default=Path("data/interim/autolabel/labels"), help="Labels root.")
    parser.add_argument("--images", type=Path, default=None, help="Images root to confirm label/image parity.")
    parser.add_argument("--classes", type=int, nargs="*", default=[0, 1], help="Allowed class ids.")
    parser.add_argument("--report", type=Path, default=Path("data/interim/qc_report.json"), help="Where to write the QC report.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_issues: List[LabelIssue] = []
    for label_file in args.labels.rglob("*.txt"):
        all_issues.extend(validate_label_file(label_file, args.classes))
    all_issues.extend(verify_image_pairs(args.labels, args.images))
    report = {
        "labels_root": str(args.labels),
        "issues": [asdict(issue) for issue in all_issues],
        "passed": not all_issues,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    if all_issues:
        print(f"QC failed with {len(all_issues)} issue(s). See {args.report}")
    else:
        print("QC passed.")


if __name__ == "__main__":
    main()
