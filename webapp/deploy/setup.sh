#!/usr/bin/env bash
# One-shot installer for the DACON dashboard on a GPU server.
# Installs: python deps + cloudflared (Cloudflare Tunnel).
#
# Usage:
#   bash webapp/deploy/setup.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
echo "[setup] project root: $ROOT"

# ---- 1. Python deps ---------------------------------------------------
echo "[setup] installing webapp Python deps..."
pip install -r "$ROOT/webapp/requirements.txt"

# ---- 2. Cloudflared ---------------------------------------------------
if ! command -v cloudflared >/dev/null 2>&1; then
    echo "[setup] cloudflared not found — installing..."
    OS="$(uname -s)"
    ARCH="$(uname -m)"
    if [[ "$OS" == "Linux" ]]; then
        if [[ "$ARCH" == "x86_64" ]]; then
            URL="https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
        elif [[ "$ARCH" == "aarch64" ]] || [[ "$ARCH" == "arm64" ]]; then
            URL="https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64"
        else
            echo "[setup] unsupported arch: $ARCH"; exit 1
        fi
        sudo curl -L "$URL" -o /usr/local/bin/cloudflared
        sudo chmod +x /usr/local/bin/cloudflared
    elif [[ "$OS" == "Darwin" ]]; then
        if command -v brew >/dev/null 2>&1; then
            brew install cloudflared
        else
            echo "[setup] install homebrew first: https://brew.sh"; exit 1
        fi
    else
        echo "[setup] unsupported OS: $OS"; exit 1
    fi
fi
cloudflared --version

# ---- 3. Secrets ------------------------------------------------------
SECRETS="$ROOT/webapp/.streamlit/secrets.toml"
if [[ ! -f "$SECRETS" ]]; then
    echo "[setup] creating secrets.toml — set a password!"
    read -rsp "Dashboard password: " PW
    echo
    cat > "$SECRETS" <<EOF
password = "$PW"
EOF
    chmod 600 "$SECRETS"
fi

echo
echo "[setup] done. Next steps:"
echo "  1. Start dashboard:    bash webapp/deploy/start_local.sh"
echo "  2. Expose via tunnel:  bash webapp/deploy/start_tunnel.sh"
echo "  3. Open the printed *.trycloudflare.com URL on any device."
