#!/bin/bash
# BeforeInstall: stop services and clear code files so CodeDeploy can install clean.
# Preserves .env, .venv/, models/, logs/, reports/ — never overwrites data or secrets.
set -euo pipefail

systemctl stop moneymaker-train.timer 2>/dev/null || true
systemctl stop moneymaker 2>/dev/null || true

APP=/opt/moneymaker

# Remove code dirs/files that CodeDeploy is about to re-install.
# Without this, CodeDeploy refuses to overwrite files it didn't place there itself.
rm -rf \
    "$APP/bots" \
    "$APP/core" \
    "$APP/envs" \
    "$APP/scripts" \
    "$APP/deploy" \
    "$APP/tests"
rm -f \
    "$APP/appspec.yml" \
    "$APP/requirements.txt" \
    "$APP/requirements.lock"

echo "Services stopped and code directories cleared."
