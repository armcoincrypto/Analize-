#!/bin/bash
#
# Install HFT Bot as a systemd service
#
# Usage: sudo ./deploy/install_service.sh
#
set -e

REPO_DIR="/root/Analize-"
SERVICE_NAME="hft-bot"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE_FILE="${SCRIPT_DIR}/systemd/hft-bot.service"
TARGET_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

echo "========================================"
echo "HFT Bot Systemd Service Installer"
echo "========================================"

# Check if running as root
if [[ $EUID -ne 0 ]]; then
    echo "ERROR: This script must be run as root (use sudo)"
    exit 1
fi

# Check if template exists
if [[ ! -f "$TEMPLATE_FILE" ]]; then
    echo "ERROR: Service template not found: $TEMPLATE_FILE"
    exit 1
fi

# Detect Python path
detect_python() {
    local candidates=(
        "${REPO_DIR}/venv/bin/python3"
        "${REPO_DIR}/venv/bin/python"
        "/usr/bin/python3"
    )

    for python_path in "${candidates[@]}"; do
        if [[ -x "$python_path" ]]; then
            echo "$python_path"
            return 0
        fi
    done

    echo "ERROR: No Python interpreter found" >&2
    return 1
}

PYTHON_PATH=$(detect_python)
if [[ $? -ne 0 ]]; then
    echo "ERROR: Could not find a valid Python interpreter"
    echo "Checked locations:"
    echo "  - ${REPO_DIR}/venv/bin/python3"
    echo "  - ${REPO_DIR}/venv/bin/python"
    echo "  - /usr/bin/python3"
    exit 1
fi

echo "Detected Python: $PYTHON_PATH"

# Verify repo directory exists
if [[ ! -d "$REPO_DIR" ]]; then
    echo "ERROR: Repository directory not found: $REPO_DIR"
    exit 1
fi

# Verify run_hft.py exists
if [[ ! -f "${REPO_DIR}/run_hft.py" ]]; then
    echo "ERROR: run_hft.py not found in $REPO_DIR"
    exit 1
fi

# Stop existing service if running
if systemctl is-active --quiet "$SERVICE_NAME" 2>/dev/null; then
    echo "Stopping existing ${SERVICE_NAME} service..."
    systemctl stop "$SERVICE_NAME"
fi

# Generate service file from template
echo "Installing systemd service to ${TARGET_FILE}..."
sed "s|__PYTHON_PATH__|${PYTHON_PATH}|g" "$TEMPLATE_FILE" > "$TARGET_FILE"

# Set proper permissions
chmod 644 "$TARGET_FILE"

echo "Reloading systemd daemon..."
systemctl daemon-reload

echo "Enabling ${SERVICE_NAME} service..."
systemctl enable "$SERVICE_NAME"

echo "Starting ${SERVICE_NAME} service..."
systemctl start "$SERVICE_NAME"

echo ""
echo "========================================"
echo "Service Status"
echo "========================================"
systemctl status "$SERVICE_NAME" --no-pager || true

echo ""
echo "========================================"
echo "Recent Logs (last 80 lines)"
echo "========================================"
journalctl -u "$SERVICE_NAME" -n 80 --no-pager 2>/dev/null || echo "No logs available yet"

echo ""
echo "========================================"
echo "Installation Complete"
echo "========================================"
echo "Service: ${SERVICE_NAME}"
echo "Python:  ${PYTHON_PATH}"
echo ""
echo "Useful commands:"
echo "  systemctl status ${SERVICE_NAME}     # Check status"
echo "  systemctl restart ${SERVICE_NAME}    # Restart service"
echo "  systemctl stop ${SERVICE_NAME}       # Stop service"
echo "  journalctl -u ${SERVICE_NAME} -f     # Follow logs"
echo "========================================"
