#!/usr/bin/env python3
"""
deploy/upload_artefacts.py
==========================
Run ONCE locally to push model artefacts to Hugging Face Hub.
The build.sh on Render will then download them at deploy time.

Usage:
  pip install huggingface-hub
  huggingface-cli login        # or set HF_TOKEN env var
  python deploy/upload_artefacts.py
"""

import os
import sys
from pathlib import Path

try:
    from huggingface_hub import HfApi, create_repo, upload_file
except ImportError:
    print("Install huggingface-hub:  pip install huggingface-hub")
    sys.exit(1)

# ─── CONFIG ────────────────────────────────────────────────────────────────
REPO_ID   = "RishabhRana37/offside-artefacts"   # ← your HF username/repo
REPO_TYPE = "dataset"                            # dataset repos have no size limit
WORKSPACE = Path(__file__).parent.parent         # project root

ARTEFACTS = [
    "catboost_model.cbm",
    "fitted_pipeline.pkl",
    "player_profiles.pkl",
    "te_smooth_maps.pkl",
]

# ─── UPLOAD ────────────────────────────────────────────────────────────────
def main():
    token = os.environ.get("HF_TOKEN")
    api   = HfApi(token=token)

    print(f"\n📦 OFF SIDE — Uploading artefacts to Hugging Face Hub")
    print(f"   Repo: {REPO_ID}  ({REPO_TYPE})")
    print()

    # Create repo if it doesn't exist
    try:
        create_repo(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            private=False,
            exist_ok=True,
            token=token,
        )
        print(f"✅ Repo ready: https://huggingface.co/{REPO_TYPE}s/{REPO_ID}")
    except Exception as e:
        print(f"⚠️  Repo creation: {e}")

    print()
    for fname in ARTEFACTS:
        fpath = WORKSPACE / fname
        if not fpath.exists():
            print(f"❌  {fname} not found — skipping")
            continue

        size_mb = fpath.stat().st_size / 1_048_576
        print(f"⬆️   {fname}  ({size_mb:.1f} MB) ...", end=" ", flush=True)

        upload_file(
            path_or_fileobj=str(fpath),
            path_in_repo=fname,
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            token=token,
        )
        print("✅")

    print()
    print("🎉  All artefacts uploaded!")
    print(f"    https://huggingface.co/datasets/{REPO_ID}")
    print()
    print("Next steps:")
    print("  1. Push this repo to GitHub (excluding large csv files)")
    print("  2. Create a new Web Service on Render.com")
    print("  3. Set Build Command:  bash build.sh")
    print("  4. Set Start Command:  streamlit run app.py --server.port $PORT --server.headless true")

if __name__ == "__main__":
    main()
