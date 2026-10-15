"""Deploy Verdantix to a free Hugging Face Docker Space in one command.

    venv\\Scripts\\python.exe scripts\\deploy_hf.py <your-hf-username>

What it does:
  1. creates the Space <username>/verdantix (Docker SDK, free CPU) if it doesn't exist
  2. uploads the code, model weights AND backend/data/cache/*.pkl (gitignored, so a
     GitHub-based deploy would miss them); files over 10 MB go up as LFS automatically
  3. writes a Space README with the header Hugging Face needs (sdk: docker, port 7860)
  4. copies GEMINI_API_KEY / GROQ_API_KEY from your .env into Space *secrets*
     (never into the uploaded files; .env itself is not uploaded)

You need a Hugging Face account and a WRITE token from https://huggingface.co/settings/tokens .
The script asks for it if you are not logged in (or set HF_TOKEN).
"""
import getpass
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
SPACE_NAME = "verdantix"

SPACE_README = """---
title: Verdantix
emoji: 🌿
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: AI digital twin of DSU Harohalli campus (CodeClash 2.0)
---

# Verdantix

Live AI digital twin of Dayananda Sagar University's Harohalli campus: energy, water and waste,
computer vision, student photo reports and a GenAI copilot. Built at CodeClash 2.0.

Code: https://github.com/princemittalr/verdantix
"""

IGNORE = ["venv/**", "datasets/**", "runs/**", ".git/**", ".env", "backend/uploads/**", "**/__pycache__/**",
          "*.log", "README.md", ".claude/**", "*.pyc", "node_modules/**",
          # keep the Space's own LFS rules; the PDF / playbook are submission files, not app files
          ".gitattributes", "*.pdf", "verdantix-playbook.html", "hardware/**"]


def env_value(key):
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8-sig").splitlines():
            if line.strip().startswith(key + "="):
                return line.split("=", 1)[1].strip()
    return os.getenv(key, "")


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python scripts/deploy_hf.py <your-hf-username>")
    user = sys.argv[1].strip()
    cache = ROOT / "backend" / "data" / "cache"
    needed = ["buildings.pkl", "rooms.pkl", "water.pkl", "occupancy.pkl", "mapping.json"]
    missing = [f for f in needed if not (cache / f).exists()]
    if missing:
        sys.exit(f"Missing {missing} in backend/data/cache. Run scripts/prepare_data.py first.")

    token = os.getenv("HF_TOKEN") or None
    api = HfApi(token=token)
    try:
        api.whoami()
    except Exception:
        token = getpass.getpass("Hugging Face WRITE token (input hidden): ").strip()
        api = HfApi(token=token)
        api.whoami()

    repo_id = f"{user}/{SPACE_NAME}"
    print(f"Creating / reusing Space {repo_id} ...")
    api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True, private=False)

    for key in ("GEMINI_API_KEY", "GROQ_API_KEY", "INGEST_TOKEN"):
        val = env_value(key)
        if val:
            api.add_space_secret(repo_id, key, val)
            print(f"  secret {key} set")
    for key in ("GEMINI_MODEL", "GROQ_MODEL"):
        val = env_value(key)
        if val:
            api.add_space_variable(repo_id, key, val)

    print("Uploading code, weights and data cache (first upload can take a few minutes) ...")
    api.upload_folder(folder_path=str(ROOT), repo_id=repo_id, repo_type="space", ignore_patterns=IGNORE,
                      commit_message="Deploy Verdantix")
    api.upload_file(path_or_fileobj=SPACE_README.encode("utf-8"), path_in_repo="README.md", repo_id=repo_id,
                    repo_type="space", commit_message="Space config")

    sub = f"{user}-{SPACE_NAME}".lower().replace("_", "-").replace(".", "-")
    print("\nDone. The Space now builds the Docker image (about 8-15 minutes the first time).")
    print(f"  Build logs : https://huggingface.co/spaces/{repo_id}?logs=build")
    print(f"  Live link  : https://{sub}.hf.space   <- put THIS in the submission form (camera + GPS work here)")


if __name__ == "__main__":
    main()
