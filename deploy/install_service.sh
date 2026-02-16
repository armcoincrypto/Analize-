#!/bin/bash
#
# Install HFT Bot as a systemd service
#
# Usage: sudo ./deploy/install_service.sh
#
set -e

REPO_DIR="/root/Analize-"
VENV_DIR="${REPO_DIR}/venv"
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

# Verify requirements.txt exists
if [[ ! -f "${REPO_DIR}/requirements.txt" ]]; then
    echo "ERROR: requirements.txt not found in $REPO_DIR"
    exit 1
fi

echo ""
echo "========================================"
echo "Step 1: Virtual Environment Setup"
echo "========================================"

# Create venv if it doesn't exist
if [[ ! -d "$VENV_DIR" ]]; then
    echo "Creating virtual environment at ${VENV_DIR}..."
    python3 -m venv "$VENV_DIR"
    if [[ $? -ne 0 ]]; then
        echo "ERROR: Failed to create virtual environment"
        echo "Make sure python3-venv is installed: apt install python3-venv"
        exit 1
    fi
    echo "Virtual environment created successfully"
else
    echo "Virtual environment already exists at ${VENV_DIR}"
fi

# Verify venv Python exists
PYTHON_PATH="${VENV_DIR}/bin/python3"
if [[ ! -x "$PYTHON_PATH" ]]; then
    echo "ERROR: Python not found in venv: $PYTHON_PATH"
    exit 1
fi

echo "Using Python: $PYTHON_PATH"
"$PYTHON_PATH" --version

echo ""
echo "========================================"
echo "Step 2: Install Dependencies"
echo "========================================"

# Upgrade pip first
echo "Upgrading pip..."
"$PYTHON_PATH" -m pip install --upgrade pip

# Install requirements
echo "Installing dependencies from requirements.txt..."
"$PYTHON_PATH" -m pip install -r "${REPO_DIR}/requirements.txt"

if [[ $? -ne 0 ]]; then
    echo "ERROR: Failed to install dependencies"
    exit 1
fi

echo "Dependencies installed successfully"

echo ""
echo "========================================"
echo "Step 3: Verify Runtime Imports"
echo "========================================"

# Critical imports to verify
CRITICAL_IMPORTS=(
    "websockets"
    "pandas"
    "numpy"
    "pydantic"
    "asyncio"
)

echo "Verifying critical imports..."
for module in "${CRITICAL_IMPORTS[@]}"; do
    echo -n "  Checking $module... "
    if "$PYTHON_PATH" -c "import $module" 2>/dev/null; then
        echo "OK"
    else
        echo "FAILED"
        echo "ERROR: Failed to import $module"
        echo "Try reinstalling: ${VENV_DIR}/bin/pip install $module"
        exit 1
    fi
done

# Verify the main HFT system can be imported
echo -n "  Checking hft_system... "
cd "$REPO_DIR"
if "$PYTHON_PATH" -c "import sys; sys.path.insert(0, '.'); from hft_system import hft_bot" 2>/dev/null; then
    echo "OK"
else
    echo "FAILED"
    echo "WARNING: Could not import hft_system.hft_bot (may work at runtime)"
fi

echo ""
echo "All critical imports verified successfully"

echo ""
echo "========================================"
echo "Step 4: Install Systemd Service"
echo "========================================"

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

echo ""
echo "========================================"
echo "Step 5: Start Service"
echo "========================================"

echo "Starting ${SERVICE_NAME} service..."
systemctl start "$SERVICE_NAME"

# Wait a moment for service to start
sleep 2

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
echo "Venv:    ${VENV_DIR}"
echo ""
echo "Useful commands:"
echo "  systemctl status ${SERVICE_NAME}     # Check status"
echo "  systemctl restart ${SERVICE_NAME}    # Restart service"
echo "  systemctl stop ${SERVICE_NAME}       # Stop service"
echo "  journalctl -u ${SERVICE_NAME} -f     # Follow logs"
echo "  journalctl -u ${SERVICE_NAME} -n 80  # Last 80 log lines"
echo "========================================"
