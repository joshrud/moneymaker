#!/bin/bash
# AfterInstall: fix permissions and install new Python dependencies
set -euo pipefail

APP=/opt/moneymaker
VENV=$APP/.venv

# Create service account if this is a fresh instance
if ! id -u moneymaker &>/dev/null; then
    useradd --system --no-create-home --shell /usr/sbin/nologin moneymaker
fi

# Create venv if it doesn't exist (e.g. first deploy or after manual rm -rf)
if [ ! -f "$VENV/bin/pip" ]; then
    python3 -m venv "$VENV"
    chown -R moneymaker:moneymaker "$VENV"
fi

# Ensure runtime dirs exist with correct ownership (never overwritten by CodeDeploy)
for dir in logs reports models; do
    mkdir -p "$APP/$dir"
    chown moneymaker:moneymaker "$APP/$dir"
done

# Fix ownership of code files
chown -R moneymaker:moneymaker "$APP/bots" "$APP/core" "$APP/envs" "$APP/scripts"

# Deploy scripts must be executable by root hooks
chmod +x "$APP/deploy"/*.sh

# Install any new or updated packages
sudo -u moneymaker "$VENV/bin/pip" install -q -r "$APP/requirements.txt"

echo "Install complete."
