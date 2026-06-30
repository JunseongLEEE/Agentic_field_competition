#!/usr/bin/env bash
# Start the Streamlit dashboard bound to 127.0.0.1:8501 only.
# External access goes through Cloudflare Tunnel — never bind to 0.0.0.0.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

# Load password from secrets.toml if env var is not already set.
if [[ -z "${DASHBOARD_PASSWORD:-}" ]] && [[ -f webapp/.streamlit/secrets.toml ]]; then
    export DASHBOARD_PASSWORD="$(python -c "import tomllib; print(tomllib.load(open('webapp/.streamlit/secrets.toml','rb'))['password'])")"
fi

if [[ -z "${DASHBOARD_PASSWORD:-}" ]]; then
    echo "[start_local] ERROR: DASHBOARD_PASSWORD not set."
    echo "  Run: bash webapp/deploy/setup.sh"
    exit 1
fi

exec streamlit run webapp/app.py
