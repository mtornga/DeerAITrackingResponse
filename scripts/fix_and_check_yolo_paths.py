from pathlib import Path

# 🧭 Update this to your dataset folder that contains data.yaml
base = Path("/Users/marktornga/Downloads/ultraYOLO")

for subset_file in ["Train.txt", "Validation.txt"]:
    txt = base / subset_file
    if not txt.exists():
        print(f"⚠️ Skipping {subset_file} (not found)")
        continue

    lines = [l.strip() for l in txt.read_text().splitlines() if l.strip()]
    fixed_lines = []
    missing_labels = []
    images_dir = base / "images"
    labels_dir = base / "labels"

    for l in lines:
        p = Path(l)
        # normalize relative path (remove any "data/" prefix)
        if "images" in p.parts:
            idx = p.parts.index("images")
            rel_path = Path(*p.parts[idx:])
        else:
            rel_path = Path("images") / p.name

        fixed_lines.append(str(rel_path))

        # Verify matching label exists
        label_rel = Path(str(rel_path).replace("images", "labels")).with_suffix(".txt")
        label_path = base / label_rel
        if not label_path.exists():
            missing_labels.append(label_rel)

    # overwrite with cleaned paths
    txt.write_text("\n".join(fixed_lines) + "\n")

    # Report results
    print(f"✅ Fixed {subset_file}: {len(fixed_lines)} entries")
    if missing_labels:
        print(f"⚠️ Missing labels ({len(missing_labels)}):")
        for m in missing_labels[:10]:
            print(f"  - {m}")
        if len(missing_labels) > 10:
            print(f"  ... and {len(missing_labels)-10} more")
    else:
        print("✅ All images have matching labels.")
