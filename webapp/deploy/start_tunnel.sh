#!/usr/bin/env bash
# Expose the local dashboard via a Cloudflare Tunnel.
#
# Mode A (default): Quick tunnel — gives a random *.trycloudflare.com URL,
#                   no Cloudflare account or domain required. URL changes each run.
# Mode B (named):   If $CF_TUNNEL_NAME is set, runs a pre-configured named tunnel
#                   on your own domain (requires `cloudflared tunnel login` once).

set -euo pipefail

PORT="${PORT:-8501}"

if [[ -n "${CF_TUNNEL_NAME:-}" ]]; then
    echo "[tunnel] running named tunnel: $CF_TUNNEL_NAME"
    exec cloudflared tunnel run "$CF_TUNNEL_NAME"
else
    echo "[tunnel] starting quick tunnel to http://localhost:$PORT"
    echo "[tunnel] Cloudflare will print a *.trycloudflare.com URL — open it on any device."
    exec cloudflared tunnel --url "http://localhost:$PORT"
fi
