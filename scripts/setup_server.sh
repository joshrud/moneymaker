#!/usr/bin/env bash
# Server provisioning script for Moneymaker on Ubuntu 24.04 LTS.
# Tested on AWS EC2 t2.micro/t3.micro (free-tier eligible).
#
# Usage (run as root on a fresh instance):
#   curl -fsSL https://raw.githubusercontent.com/joshrud/moneymaker/main/scripts/setup_server.sh | sudo bash
#   -- or --
#   sudo bash scripts/setup_server.sh
#
# After this script completes you still need to:
#   1. Edit /opt/moneymaker/.env with your real API keys
#   2. Run the initial model training (instructions printed at the end)

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────────
GITHUB_REPO_URL="https://github.com/joshrud/moneymaker.git"
INSTALL_DIR="/opt/moneymaker"
SERVICE_USER="moneymaker"
# Ubuntu 24.04 ships Python 3.12 — no PPA needed
PYTHON="python3.12"

# ── Guards ─────────────────────────────────────────────────────────────────────
if [[ $EUID -ne 0 ]]; then
  echo "ERROR: run this script as root (sudo bash setup_server.sh)" >&2
  exit 1
fi

UBUNTU_VER=$(lsb_release -rs 2>/dev/null || echo "unknown")
if [[ "$UBUNTU_VER" != "24.04" && "$UBUNTU_VER" != "26.04" ]]; then
  echo "WARNING: this script targets Ubuntu 24.04. Detected: $UBUNTU_VER — continuing anyway..."
fi

echo "=== Moneymaker server setup ==="
echo "Install dir : $INSTALL_DIR"
echo "Repo        : $GITHUB_REPO_URL"
echo "Ubuntu      : $UBUNTU_VER"
echo ""

# ── 1. System packages ─────────────────────────────────────────────────────────
echo "[1/7] Installing system packages..."
apt-get update -qq
apt-get install -y -qq git curl python3.12 python3.12-venv python3.12-dev

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
  sudo -u "$SERVICE_USER" "$INSTALL_DIR/.venv/bin/pip" install -q --upgrade pip
  sudo -u "$SERVICE_USER" "$INSTALL_DIR/.venv/bin/pip" install -q -r "$LOCK_FILE"
else
  echo "      requirements.lock not found — falling back to requirements.txt..."
  sudo -u "$SERVICE_USER" "$INSTALL_DIR/.venv/bin/pip" install -q --upgrade pip
  sudo -u "$SERVICE_USER" "$INSTALL_DIR/.venv/bin/pip" install -q -r "$REQS_FILE"
fi

# Verify torch installed; if missing, install CPU-only build (saves ~1 GB vs CUDA default)
TORCH_VER=$("$INSTALL_DIR/.venv/bin/python" -c "import torch; print(torch.__version__)" 2>/dev/null || echo "")
if [[ -z "$TORCH_VER" ]]; then
  echo "      torch not found — installing CPU-only build..."
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

# ── 6. Log and model directories ──────────────────────────────────────────────
echo "[6/7] Creating log and model directories..."
mkdir -p "$INSTALL_DIR/logs" "$INSTALL_DIR/models"
chown -R "$SERVICE_USER:$SERVICE_USER" "$INSTALL_DIR/logs" "$INSTALL_DIR/models"

# ── 7. systemd service ─────────────────────────────────────────────────────────
echo "[7/7] Installing systemd service..."
cp "$INSTALL_DIR/deploy/moneymaker.service" /etc/systemd/system/moneymaker.service
cp "$INSTALL_DIR/deploy/moneymaker-train.service" /etc/systemd/system/moneymaker-train.service
cp "$INSTALL_DIR/deploy/moneymaker-train.timer"   /etc/systemd/system/moneymaker-train.timer
systemctl daemon-reload
systemctl enable moneymaker
systemctl enable --now moneymaker-train.timer
echo "      Services installed. Bot service enabled (start after filling .env)."
echo "      Training timer enabled and running (next fire: $(systemctl show moneymaker-train.timer --property=NextElapseUSecRealtime --value))"

# ── Done ───────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " Setup complete. Next steps:"
echo "============================================================"
echo ""
echo " 1. Fill in API keys:"
echo "    nano $ENV_FILE"
echo ""
echo " 2. Train the Bot 2 SAC model (run once, ~30-60 min):"
echo "    sudo -u $SERVICE_USER $INSTALL_DIR/.venv/bin/python \\"
echo "      $INSTALL_DIR/scripts/train_rl_models.py --bot2-only"
echo ""
echo " 3. Start the bot service:"
echo "    systemctl start moneymaker"
echo "    systemctl status moneymaker"
echo ""
echo " 4. Watch live logs:"
echo "    tail -f $INSTALL_DIR/logs/service.log"
echo "    journalctl -u moneymaker -f    # systemd journal (includes crashes)"
echo ""
echo " 5. Add weekly model retraining — paste into: crontab -e -u $SERVICE_USER"
echo "    0 4 * * 0 $INSTALL_DIR/.venv/bin/python $INSTALL_DIR/scripts/train_rl_models.py --bot2-only >> $INSTALL_DIR/logs/retrain.log 2>&1"
echo ""
echo " 6. (Optional) Add log rotation — copy to /etc/logrotate.d/moneymaker:"
echo "    $INSTALL_DIR/deploy/moneymaker-logrotate"
echo "============================================================"
