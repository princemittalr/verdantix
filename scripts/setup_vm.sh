#!/usr/bin/env bash
# One-shot setup of the Verdantix backend on a fresh Ubuntu 22.04/24.04 VM (Azure, Oracle, GCP...).
#
#   bash setup_vm.sh <public-domain>
#   e.g. bash setup_vm.sh verdantix.centralindia.cloudapp.azure.com
#
# Before running: copy the data cache from your laptop to ~/cache on the VM:
#   scp -r backend\data\cache azureuser@<vm-ip>:~/cache
#
# Result: uvicorn runs as a service (restarts on crash/reboot) behind Caddy,
# which gets a free HTTPS certificate for <public-domain>. Ports 80 and 443 must be open.
set -euo pipefail

DOMAIN="${1:?Usage: bash setup_vm.sh <public-domain>}"
REPO_URL="${REPO_URL:-https://github.com/princemittalr/verdantix.git}"
APP="$HOME/verdantix"

echo "==> System packages"
sudo apt-get update -y
sudo apt-get install -y python3 python3-venv python3-pip git libgl1 libglib2.0-0 \
  debian-keyring debian-archive-keyring apt-transport-https curl gnupg

echo "==> Code"
if [ -d "$APP/.git" ]; then git -C "$APP" pull --ff-only; else git clone --depth 1 "$REPO_URL" "$APP"; fi

echo "==> Data cache"
mkdir -p "$APP/backend/data/cache"
if [ -d "$HOME/cache" ]; then cp -r "$HOME/cache/." "$APP/backend/data/cache/"; fi
for f in buildings.pkl rooms.pkl water.pkl occupancy.pkl mapping.json; do
  [ -f "$APP/backend/data/cache/$f" ] || { echo "MISSING backend/data/cache/$f - scp the cache folder first (see top of this file)"; exit 1; }
done

echo "==> Python environment (CPU-only torch; takes 5-10 min)"
cd "$APP"
python3 -m venv venv
venv/bin/pip install --upgrade pip wheel
venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
venv/bin/pip install -r requirements.txt
venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('openai/clip-vit-base-patch32', allow_patterns=['*.json','*.txt','model.safetensors'])"

echo "==> API keys (.env)"
if [ ! -f "$APP/.env" ]; then
  read -rsp "GEMINI_API_KEY (Enter to skip): " GK; echo
  read -rsp "GROQ_API_KEY (Enter to skip): " QK; echo
  printf 'GEMINI_API_KEY=%s\nGROQ_API_KEY=%s\nGEMINI_MODEL=gemini-3.8-flash\nGROQ_MODEL=openai/gpt-oss-120b\n' "$GK" "$QK" > "$APP/.env"
  chmod 600 "$APP/.env"
fi
mkdir -p "$APP/backend/uploads"

echo "==> Service"
sudo tee /etc/systemd/system/verdantix.service >/dev/null <<EOF
[Unit]
Description=Verdantix digital twin API
After=network-online.target

[Service]
User=$USER
WorkingDirectory=$APP
Environment=PYTHONUNBUFFERED=1
ExecStart=$APP/venv/bin/python -m uvicorn app:app --app-dir backend --host 127.0.0.1 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable --now verdantix

echo "==> Caddy (automatic HTTPS + WebSockets)"
if ! command -v caddy >/dev/null; then
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor --yes -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list >/dev/null
  sudo apt-get update -y && sudo apt-get install -y caddy
fi
printf '%s {\n\tencode gzip\n\treverse_proxy 127.0.0.1:8000\n}\n' "$DOMAIN" | sudo tee /etc/caddy/Caddyfile >/dev/null
sudo systemctl restart caddy

echo "==> Waiting for the twin to start (model training takes ~30-60 s)"
for i in $(seq 1 60); do
  if curl -fs -o /dev/null http://127.0.0.1:8000/api/state; then break; fi
  sleep 3
done
curl -fs -o /dev/null http://127.0.0.1:8000/api/state && echo "Backend OK" || { echo "Backend not up yet. Check: sudo journalctl -u verdantix -n 50"; exit 1; }

echo
echo "Done. Live backend: https://$DOMAIN   (test: https://$DOMAIN/api/state)"
echo "Set window.VX_API = \"https://$DOMAIN\"; in frontend/config.js and push to update Vercel."
echo "Logs: sudo journalctl -u verdantix -f      Update code: git -C $APP pull && sudo systemctl restart verdantix"
