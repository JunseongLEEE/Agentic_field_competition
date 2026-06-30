# DACON Competition Dashboard

A Streamlit web UI for recording DACON leaderboard scores and watching the
CV→LB correlation in real time. Replaces typing
`/submit-result exp_001 0.82` in Claude Code with a form you can fill out
from any browser (including a phone) while looking at the DACON site.

All writes go through the existing scripts (`track_submission.py`,
`cv_lb_correlation.py`, `build_digest.py`), so the dashboard and the slash
commands stay in perfect sync — they both edit the same files.

---

## Quick Start (local-only)

```bash
# 1. Install deps + create password
bash webapp/deploy/setup.sh

# 2. Start dashboard on 127.0.0.1:8501
bash webapp/deploy/start_local.sh

# 3. Open http://127.0.0.1:8501 in your browser
```

---

## External Access via Cloudflare Tunnel

The dashboard binds to `127.0.0.1` only. To reach it from your phone or
laptop on another network, expose it through a Cloudflare Tunnel. No port
forwarding, no firewall changes — the tunnel makes an outbound connection
to Cloudflare's edge.

### Mode A — Quick tunnel (no domain needed)

```bash
# Terminal 1: dashboard
bash webapp/deploy/start_local.sh

# Terminal 2: tunnel
bash webapp/deploy/start_tunnel.sh
```

Cloudflare will print something like:
```
https://random-words-12345.trycloudflare.com
```

Open that URL on any device. URL changes on every restart.

### Mode B — Named tunnel (stable URL on your domain)

Requires a domain managed by Cloudflare (free).

```bash
# One-time
cloudflared tunnel login
cloudflared tunnel create dacon-dashboard
cloudflared tunnel route dns dacon-dashboard dacon.your-domain.com

# Then start with:
CF_TUNNEL_NAME=dacon-dashboard bash webapp/deploy/start_tunnel.sh
```

---

## Auto-start on boot (Linux GPU server)

Edit `webapp/deploy/dashboard.service` and `webapp/deploy/tunnel.service`:
- Set `User=` to your username
- Set `WorkingDirectory=` to the absolute path of the repo

Then:
```bash
sudo cp webapp/deploy/dashboard.service /etc/systemd/system/
sudo cp webapp/deploy/tunnel.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now dashboard tunnel

# Logs
journalctl -u dashboard -f
journalctl -u tunnel -f
```

---

## Authentication

Single-password gate (you set it in `.streamlit/secrets.toml`).

Password options, in priority order:
1. `DASHBOARD_PASSWORD` env var
2. `webapp/.streamlit/secrets.toml` → `password = "..."`

`.streamlit/secrets.toml` is `.gitignore`d — never commit it.

### Stronger auth (optional upgrade)

If you later want SSO instead of a shared password, add
[Cloudflare Access](https://developers.cloudflare.com/cloudflare-one/) on
top of the named tunnel — auth happens at the Cloudflare edge, before the
request reaches your GPU server. Free for ≤50 users.

---

## What the dashboard shows

- **Header**: D-day countdown, daily quota, best CV, best LB, CV→LB trust level
- **Submission form**: dropdown of packaged experiments + pre-filled CV + predicted LB + PI
- **Recent submissions**: last 15 entries from `competition_meta.yaml.submissions_log`
- **CV vs LB scatter**: every successful submission plotted
- **Family bracket**: leader / explored / dropped status per model family (from `orchestrator_state.json`)
- **Recent insights**: last 5 entries from `logs/insights.jsonl`

## What the dashboard does NOT do

- Does not upload to DACON for you (manual upload only — the competition
  framework explicitly forbids automation here).
- Does not run experiments (use `/auto` in Claude Code for that).
- Does not edit experiment code (read-only display + write to
  `submissions_log` / correlation files only).

## Safety

- Every write backs up `competition_meta.yaml` to `logs/backups/`
  (keeps last 30 backups).
- Duplicate submission for the same experiment shows a warning before write.
- LB values clamped to `[0, 1]`.
- Login session times out when you close the tab.

---

## File map

```
webapp/
├── app.py                          # Streamlit entry point
├── requirements.txt
├── README.md
├── .streamlit/
│   ├── config.toml                 # bind to 127.0.0.1:8501, dark theme
│   └── secrets.toml.example        # copy to secrets.toml, set password
└── deploy/
    ├── setup.sh                    # one-shot installer
    ├── start_local.sh              # run Streamlit
    ├── start_tunnel.sh             # run cloudflared
    ├── dashboard.service           # systemd unit for Streamlit
    └── tunnel.service              # systemd unit for cloudflared
```
