import json
from pathlib import Path

import streamlit as st
import yaml

DATASETS_ROOT = Path("data/datasets")
MODELS_ROOT = Path("models")
THRESHOLDS = Path("configs/eval_thresholds.yaml")


def list_subdirs(root: Path) -> list[str]:
    if not root.exists():
        return []
    return sorted([p.name for p in root.iterdir() if p.is_dir()])


def load_metrics(model_name: str) -> dict:
    metrics_path = MODELS_ROOT / model_name / "metrics.json"
    if not metrics_path.exists():
        return {}
    return json.loads(metrics_path.read_text())


def main() -> None:
    st.set_page_config(page_title="Deer Vision – Detector Dashboard", layout="wide")
    st.title("Deer Vision Detector Dashboard")

    cols = st.columns(2)
    with cols[0]:
        dataset = st.selectbox("Dataset Version", list_subdirs(DATASETS_ROOT) or ["(none)"])
    with cols[1]:
        model = st.selectbox("Model Run", list_subdirs(MODELS_ROOT) or ["(none)"])

    st.subheader("Quick Actions")
    st.code("make frames && make autolabel && make qc && make split", language="bash")
    st.code(f"python scripts/train.py --data configs/data_deer.yaml --name {model}", language="bash")
    st.code(f"python scripts/evaluate.py --model models/{model}/weights/best.pt --eval-packs data/eval", language="bash")

    metrics = load_metrics(model)
    thresholds = yaml.safe_load(THRESHOLDS.read_text()) if THRESHOLDS.exists() else {}

    st.subheader("Evaluation Metrics")
    if not metrics:
        st.info("No metrics found for the selected model.")
    else:
        cols = st.columns(3)
        cols[0].metric("mAP@0.5", f"{metrics.get('map50', 0):.3f}", delta=f"min {thresholds.get('map50_min', 0):.2f}")
        cols[1].metric("Recall@0.5", f"{metrics.get('recall', 0):.3f}", delta=f"min {thresholds.get('recall_min', 0):.2f}")
        cols[2].metric("Latency p95 (ms)", f"{metrics.get('latency_ms_p95', 0):.1f}", delta=f"max {thresholds.get('latency_ms_p95_max', 0):.1f}")
        st.json(metrics.get("fp_per_min", {}))

    st.subheader("Frozen Eval Packs")
    for pack in Path("data/eval").iterdir():
        if not pack.is_dir():
            continue
        st.markdown(f"- **{pack.name}** – labels: `{pack / 'labels'}`")

    st.subheader("Notes")
    st.write("Use `scripts/visualize_errors.py` to render FN/FP panels. Upload curated clips to S3 for DVC tracking.")


if __name__ == "__main__":
    main()
