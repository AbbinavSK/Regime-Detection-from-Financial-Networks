import argparse

import pandas as pd

from regime_networks import config
from regime_networks.data import MANIFEST_NAME, file_sha256, missing_data_message


# One manifest row per CSV in data/: checksum, size and shape, so a published dataset can be verified byte for byte.
def build_manifest():
    rows = []

    for path in sorted(config.DATA_DIR.glob("*.csv")):
        if path.name == MANIFEST_NAME:
            continue

        frame = pd.read_csv(path)
        rows.append({"file": path.name, "sha256": file_sha256(path), "bytes": path.stat().st_size, "rows": len(frame), "columns": frame.shape[1]})

    return pd.DataFrame(rows)


# Compare the current data/ files with a saved manifest; returns the listed files that are missing and those whose checksums differ.
def verify(manifest_path):
    expected = pd.read_csv(manifest_path).set_index("file")["sha256"]
    missing = [name for name in expected.index if not (config.DATA_DIR / name).exists()]
    return missing, [name for name, digest in expected.items() if name not in missing and file_sha256(config.DATA_DIR / name) != digest]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Write data/MANIFEST.csv (SHA-256, size, shape of every data CSV) or verify data/ against it.")
    parser.add_argument("--verify", action="store_true", help="Check data/ against the existing MANIFEST.csv instead of writing a new one")
    args = parser.parse_args()
    manifest_path = config.DATA_DIR / MANIFEST_NAME

    if args.verify:
        missing, changed = verify(manifest_path)

        if missing:
            print(missing_data_message([config.DATA_DIR / name for name in missing]))

        if changed:
            print(f"{len(changed)} files differ from {manifest_path}, so they are not the data the paper used: {changed}")

        if not missing and not changed:
            print(f"all files match {manifest_path}")

        raise SystemExit(1 if missing or changed else 0)

    manifest = build_manifest()
    manifest.to_csv(manifest_path, index=False, lineterminator="\n")
    print(f"wrote {manifest_path} ({len(manifest)} files)")
