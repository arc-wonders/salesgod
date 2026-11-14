#!/usr/bin/env bash
# One-time setup of the salesgod voice stack on a GPU VM (Ubuntu 22.04). Re-runnable.
set -euo pipefail
id salesgod >/dev/null 2>&1 || sudo useradd -r -m -d /opt/salesgod -s /bin/bash salesgod
sudo mkdir -p /opt/salesgod/app /opt/salesgod/hf && sudo chown -R salesgod:salesgod /opt/salesgod

if ! command -v caddy >/dev/null; then
  sudo apt-get update -qq && sudo apt-get install -y -qq debian-keyring debian-archive-keyring apt-transport-https curl
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list >/dev/null
  sudo apt-get update -qq && sudo apt-get install -y -qq caddy
fi
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sudo env UV_INSTALL_DIR=/usr/local/bin sh; }

sudo -u salesgod bash -c '
set -e; cd /opt/salesgod
[ -d venv ] || uv venv -q -p 3.12 venv
uv pip install -q -p venv/bin/python -r app/deploy/voice/requirements.txt -e app
[ -d venv-vllm ] || uv venv -q -p 3.12 venv-vllm
uv pip install -q -p venv-vllm/bin/python "vllm==0.10.1.1" "transformers==4.55.4"
'
sudo cp /opt/salesgod/app/deploy/voice/Caddyfile /etc/caddy/Caddyfile
sudo cp /opt/salesgod/app/deploy/voice/salesgod-*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now salesgod-llm salesgod-voice
sudo systemctl reload caddy || sudo systemctl restart caddy
