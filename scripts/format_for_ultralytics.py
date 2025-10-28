from pathlib import Path

base = Path("/Users/marktornga/Downloads/project_1_dataset_2025_10_23_19_24_50_ultralytics yolo detection 1.0")  # <-- update this
for file in ["Train.txt", "Validation.txt"]:
    txt = base / file
    if not txt.exists():
        print(f"Skipping {file} (not found)")
        continue

    lines = [l.strip() for l in txt.read_text().splitlines() if l.strip()]
    new_lines = []
    for l in lines:
        p = Path(l)
        # find the part starting at "images/"
        if "images" in p.parts:
            idx = p.parts.index("images")
            rel = Path(*p.parts[idx:])
            new_lines.append(str(rel))
        else:
            new_lines.append(str(p))
    txt.write_text("\n".join(new_lines) + "\n")
    print(f"✅ Fixed {file} ({len(lines)} lines)")
