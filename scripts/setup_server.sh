#!/usr/bin/env bash
# Server provisioning script for Moneymaker on Ubuntu 22.04.
# Run as root on a fresh DigitalOcean Droplet (or AWS Lightsail) after
# pushing your repo to GitHub.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/YOUR_ORG/moneymaker/main/scripts/setup_server.sh | sudo bash
#   -- or --
#   sudo bash scripts/setup_server.sh
#
# After this script completes you still need to:
#   1. Edit /opt/moneymaker/.env with your real API keys
#   2. Run the initial model training (instructions printed at the end)

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────────
GITHUB_REPO_URL="https://github.com/YOUR_ORG/moneymaker.git"   # <── set this
INSTALL_DIR="/opt/moneymaker"
SERVICE_USER="moneymaker"
PYTHON="python3.11"

# ── Guards ─────────────────────────────────────────────────────────────────────
if [[ $EUID -ne 0 ]]; then
  echo "ERROR: run this script as root (sudo bash setup_server.sh)" >&2
  exit 1
fi

if [[ "$GITHUB_REPO_URL" == *"YOUR_ORG"* ]]; then
  echo "ERROR: set GITHUB_REPO_URL at the top of this script before running." >&2
  exit 1
fi

echo "=== Moneymaker server setup ==="
echo "Install dir : $INSTALL_DIR"
echo "Repo        : $GITHUB_REPO_URL"
echo ""

# ── 1. System packages ─────────────────────────────────────────────────────────
echo "[1/7] Installing system packages..."
apt-get update -qq
apt-get install -y -qq software-properties-common git curl

# Python 3.11 via deadsnakes PPA (Ubuntu 22.04 ships 3.10)
add-apt-repository -y ppa:deadsnakes/ppa
apt-get update -qq
apt-get install -y -qq python3.11 python3.11-venv python3.11-dev

echo "      Python: $($PYTHON --version)"

# ── 2. Dedicated system user ───────────────────────────────────────────────────
echo "[2/7] Creating service user '$SERVICE_USER'..."
if ! id "$SERVICE_USER" &>/dev/null; then
  useradd --system --shell /usr/sbin/nologin --home-dir "$INSTALL_DIR" "$SERVICE_USER"
fi

# ── 3. Clone repo ─────────────────────────────────────────────────────────────
echo "[3/7] Cloning repository..."
if [[ -d "$INSTALL_DIR/.git" ]]; then
  echo "      Repo already cloned — pulling latest..."
  git -C "$INSTALL_DIR" pull --ff-only
else
  git clone "$GITHUB_REPO_URL" "$INSTALL_DIR"
fi
chown -R "$SERVICE_USER:$SERVICE_USER" "$INSTALL_DIR"

# ── 4. Python virtual environment ─────────────────────────────────────────────
echo "[4/7] Creating .venv and installing dependencies..."
if [[ ! -d "$INSTALL_DIR/.venv" ]]; then
  sudo -u "$SERVICE_USER" $PYTHON -m venv "$INSTALL_DIR/.venv"
fi

LOCK_FILE="$INSTALL_DIR/requirements.lock"
REQS_FILE="$INSTALL_DIR/requirements.txt"

if [[ -f "$LOCK_FILE" ]]; then
  echo "      Installing from requirements.lock (pinned)..."
  sudo -u "$SERVICE_USER" "$INSTALL_DIR/.venv/bin/pip" install -q -r "$LOCK_FILE"
else
  echo "      requirements.lock not found — falling back to requirements.txt..."
  sudo -u "$SERVICE_USER" "$INSTALL_DIR/.venv/bin/pip" install -q -r "$REQS_FILE"
fi

# Install CPU-only torch if not already present (saves ~1 GB vs default CUDA build)
# Comment this block out if you want the default build.
TORCH_VER=$("$INSTALL_DIR/.venv/bin/python" -c "import torch; print(torch.__version__)" 2>/dev/null || echo "")
if [[ -z "$TORCH_VER" ]]; then
  echo "      Installing PyTorch (CPU-only)..."
  sudo -u "$SERVICE_USER" "$INSTALL_DIR/.venv/bin/pip" install -q \
    torch --index-url https://download.pytorch.org/whl/cpu
fi

# ── 5. Secrets (.env) ─────────────────────────────────────────────────────────
echo "[5/7] Setting up .env..."
ENV_FILE="$INSTALL_DIR/.env"
if [[ ! -f "$ENV_FILE" ]]; then
  cat > "$ENV_FILE" <<'EOF'
ALPACA_API_KEY=REPLACE_ME
ALPACA_SECRET_KEY=REPLACE_ME
ANTHROPIC_API_KEY=REPLACE_ME
EOF
  chown "$SERVICE_USER:$SERVICE_USER" "$ENV_FILE"
  chmod 600 "$ENV_FILE"
  echo "      Created .env template at $ENV_FILE"
  echo "      *** ACTION REQUIRED: fill in your API keys before starting the service ***"
else
  echo "      .env already exists — skipping."
fi

# ── 6. Log directory ───────────────────────────────────────────────────────────
echo "[6/7] Creating log directory..."
mkdir -p "$INSTALL_DIR/logs" "$INSTALL_DIR/models"
chown -R "$SERVICE_USER:$SERVICE_USER" "$INSTALL_DIR/logs" "$INSTALL_DIR/models"

# ── 7. systemd service ─────────────────────────────────────────────────────────
echo "[7/7] Installing systemd service..."
cp "$INSTALL_DIR/deploy/moneymaker.service" /etc/systemd/system/moneymaker.service
systemctl daemon-reload
systemctl enable moneymaker
echo "      Service installed and enabled (not started yet — fill in .env first)."

# ── Done ───────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " Setup complete. Next steps:"
echo "============================================================"
echo ""
echo " 1. Fill in API keys:"
echo "    nano $ENV_FILE"
echo ""
echo " 2. Train the Bot 2 SAC model (run once before starting):"
echo "    sudo -u $SERVICE_USER $INSTALL_DIR/.venv/bin/python \\"
echo "      $INSTALL_DIR/scripts/train_rl_models.py --bot2-only"
echo "    (~60 s data fetch + ~30 min training)"
echo ""
echo " 3. Start the bot service:"
echo "    systemctl start moneymaker"
echo "    systemctl status moneymaker"
echo ""
echo " 4. Watch live logs:"
echo "    tail -f $INSTALL_DIR/logs/service.log"
echo ""
echo " 5. Add weekly model retraining — paste this into: crontab -e -u $SERVICE_USER"
echo "    0 4 * * 0 $INSTALL_DIR/.venv/bin/python $INSTALL_DIR/scripts/train_rl_models.py --bot2-only >> $INSTALL_DIR/logs/retrain.log 2>&1"
echo ""
echo " 6. (Optional) Add log rotation to prevent unbounded growth:"
echo "    Copy deploy/moneymaker-logrotate to /etc/logrotate.d/moneymaker"
echo "============================================================"
