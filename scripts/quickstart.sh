#!/usr/bin/env bash
# ============================================================================
# LatentGate Quickstart
# ============================================================================
# One-command setup: starts Ollama, pulls models, runs the API, and launches
# the demo website. Works on Linux, macOS, and Windows (Git Bash / WSL).
#
# Usage:
#   chmod +x scripts/quickstart.sh
#   ./scripts/quickstart.sh
#
# Options:
#   --no-pull         Skip pulling Ollama models (use already-pulled ones)
#   --no-website      Don't start the website dev server
#   --port PORT       API server port (default: 8000)
#   --model MODEL     Ollama text model (default: llama3:8b)
#   --help            Show this help
# ============================================================================

set -euo pipefail

# ---- Colors ----
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ---- Defaults ----
PULL_MODELS=true
START_WEBSITE=true
API_PORT=8000
OLLAMA_CONTAINER="latentgate-ollama"
TEXT_MODEL="llama3:8b"
VISION_MODEL="llava:7b"
FAST_MODEL="phi3:mini"
SMART_MODEL="qwen2:7b"
EMBEDDING_MODEL="nomic-embed-text"

# ---- Parse args ----
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-pull)    PULL_MODELS=false; shift ;;
    --no-website) START_WEBSITE=false; shift ;;
    --port)       API_PORT="$2"; shift 2 ;;
    --model)      TEXT_MODEL="$2"; shift 2 ;;
    --help)       head -n 30 "$0"; exit 0 ;;
    *)            echo "Unknown option: $1"; exit 1 ;;
  esac
done

# ---- Prerequisites ----
echo -e "${CYAN}═══════════════════════════════════════════════${NC}"
echo -e "${CYAN}  LatentGate Quickstart${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════${NC}"
echo ""

check_cmd() {
  if ! command -v "$1" &>/dev/null; then
    echo -e "${RED}✗ $1 is not installed.${NC}"
    echo "  Install: $2"
    exit 1
  fi
  echo -e "${GREEN}✓ $1 found${NC}"
}

echo -e "${YELLOW}Checking prerequisites...${NC}"
check_cmd "python3" "brew install python3 / apt install python3 / python.org"
check_cmd "docker" "https://docs.docker.com/get-docker/"

# Verify Docker is running
if ! docker info &>/dev/null; then
  echo -e "${RED}✗ Docker is installed but not running.${NC}"
  echo "  Start Docker Desktop or the docker daemon first."
  exit 1
fi

# Check Python version (>= 3.10)
if python3 -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" 2>/dev/null; then
  PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
  echo -e "${GREEN}✓ Python $PY_VERSION (>= 3.10)${NC}"
else
  echo -e "${RED}✗ Python version < 3.10 (install Python 3.10 or later)${NC}"
  exit 1
fi

# Check for Node.js (optional, only for website)
if $START_WEBSITE; then
  check_cmd "node" "brew install node / apt install nodejs / nodejs.org"
  NODE_VERSION=$(node -v | sed 's/v//')
  echo -e "${GREEN}  Node.js $NODE_VERSION${NC}"
fi

echo ""

# ---- Start Ollama ----
echo -e "${YELLOW}[1/4] Starting Ollama via Docker...${NC}"

# Check if Ollama is already running
if curl -sf http://localhost:11434/api/tags &>/dev/null; then
  echo -e "${GREEN}✓ Ollama is already running${NC}"
else
  # Start Ollama container (use unique name to avoid conflicts)
  if docker inspect "$OLLAMA_CONTAINER" &>/dev/null; then
    echo -e "  Container '$OLLAMA_CONTAINER' already exists, starting it..."
    docker start "$OLLAMA_CONTAINER" &>/dev/null
  else
    docker run -d --name "$OLLAMA_CONTAINER" \
      -p 11434:11434 \
      -v ollama-data:/root/.ollama \
      ollama/ollama:latest 2>&1
  fi

  echo -e "  Waiting for Ollama to be ready..."
  for i in $(seq 1 30); do
    if curl -sf http://localhost:11434/api/tags &>/dev/null; then
      echo -e "${GREEN}✓ Ollama is ready!${NC}"
      break
    fi
    sleep 2
  done
  if ! curl -sf http://localhost:11434/api/tags &>/dev/null; then
    echo -e "${RED}✗ Ollama failed to start. Check 'docker logs $OLLAMA_CONTAINER'${NC}"
    exit 1
  fi
fi
echo ""

# ---- Pull Models ----
echo -e "${YELLOW}[2/4] Pulling Ollama models...${NC}"

pull_model() {
  local model="$1"
  local label="$2"
  if curl -sf "http://localhost:11434/api/tags" | python3 -c "import sys,json; d=json.load(sys.stdin); sys.exit(0 if any('$model' in m['name'] for m in d.get('models',[])) else 1)" 2>/dev/null; then
    echo -e "${GREEN}  ✓ $model ($label) already pulled${NC}"
  else
    echo -e "  → Pulling $model ($label)..."
    docker exec "$OLLAMA_CONTAINER" ollama pull "$model" 2>&1 | tail -1
    echo -e "${GREEN}  ✓ $model pulled${NC}"
  fi
}

if $PULL_MODELS; then
  pull_model "$VISION_MODEL"  "vision model (image compression)"
  pull_model "$FAST_MODEL"    "fast text compression"
  pull_model "$SMART_MODEL"   "smart text compression"
  pull_model "$EMBEDDING_MODEL" "embeddings"
  pull_model "$TEXT_MODEL"    "fallback / offline mode"
else
  echo -e "  Skipping (--no-pull)"
fi
echo ""

# ---- Install & Start API ----
echo -e "${YELLOW}[3/4] Starting LatentGate API server...${NC}"

# Install the package in dev mode (within a venv if available)
VENV_DIR=".latentgate-venv"
if [[ ! -d "$VENV_DIR" ]]; then
  echo -e "  Creating virtual environment..."
  python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate" 2>/dev/null || source "$VENV_DIR/Scripts/activate" 2>/dev/null

echo -e "  Installing LatentGate..."
pip install -e ".[api]" --quiet 2>&1 | tail -1

# Kill any existing instance on the port
if command -v lsof &>/dev/null; then
  kill $(lsof -ti :$API_PORT) 2>/dev/null || true
elif command -v fuser &>/dev/null; then
  fuser -k "${API_PORT}/tcp" 2>/dev/null || true
fi

# Start API server in background
LATENTGATE_HOST=0.0.0.0 \
LATENTGATE_PORT=$API_PORT \
LATENTGATE_LOG_LEVEL=INFO \
LATENTGATE_REMOTE_PROVIDER=ollama \
LATENTGATE_REMOTE_MODEL=$TEXT_MODEL \
LATENTGATE_VISION_MODEL=$VISION_MODEL \
LATENTGATE_TEXT_FAST_MODEL=$FAST_MODEL \
LATENTGATE_TEXT_SMART_MODEL=$SMART_MODEL \
nohup python -m latent_gate.api_server > /tmp/latentgate-api.log 2>&1 &
API_PID=$!
echo -e "${GREEN}  API server starting (PID: $API_PID)...${NC}"

# Wait for API to be ready
for i in $(seq 1 15); do
  if curl -sf "http://localhost:$API_PORT/health" &>/dev/null; then
    echo -e "${GREEN}✓ API server is ready at http://localhost:$API_PORT${NC}"
    break
  fi
  sleep 1
done
if ! curl -sf "http://localhost:$API_PORT/health" &>/dev/null; then
  echo -e "${RED}✗ API server failed to start. Check /tmp/latentgate-api.log${NC}"
  exit 1
fi
echo ""

# ---- Start Website ----
if $START_WEBSITE; then
  echo -e "${YELLOW}[4/4] Starting demo website...${NC}"
  cd website
  npm install --silent 2>/dev/null
  VITE_API_URL="http://localhost:$API_PORT" \
  npx vite --host 0.0.0.0 --port 5173 &
  WEB_PID=$!
  echo -e "${GREEN}  Website starting (PID: $WEB_PID)...${NC}"
  cd ..
  sleep 3
  echo -e "${GREEN}✓ Website at http://localhost:5173${NC}"
else
  echo -e "${YELLOW}[4/4] Skipping website (--no-website)${NC}"
fi

echo ""
echo -e "${CYAN}═══════════════════════════════════════════════${NC}"
echo -e "${CYAN}  ✓ LatentGate is running!${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════${NC}"
echo ""
echo -e "  API:        ${GREEN}http://localhost:$API_PORT${NC}"
echo -e "  Health:     ${GREEN}http://localhost:$API_PORT/health${NC}"
echo -e "  Compress:   ${GREEN}http://localhost:$API_PORT/compress${NC} (POST)"
echo -e "  Metrics:    ${GREEN}http://localhost:$API_PORT/metrics${NC}"
echo -e "  Ollama:     ${GREEN}http://localhost:11434${NC}"
echo -e "  API Doc:    ${GREEN}http://localhost:$API_PORT/docs${NC}"
echo ""
if $START_WEBSITE; then
  echo -e "  Website:    ${GREEN}http://localhost:5173${NC}"
  echo ""
fi
echo -e "  Try it:"
echo -e '    curl http://localhost:'$API_PORT'/compress \\'
echo -e '      -H "Content-Type: application/json" \\'
echo -e '      -d '\''{"text": "Write a Python script that reads a CSV file and calculates the average of each column."}'\'
echo ""
echo -e "${YELLOW}  Press Ctrl+C to stop all services${NC}"
echo ""

# ---- Cleanup on exit ----
cleanup() {
  echo ""
  echo -e "${YELLOW}Shutting down...${NC}"
  kill $API_PID 2>/dev/null || true
  if [[ -n "${WEB_PID:-}" ]]; then
    kill $WEB_PID 2>/dev/null || true
  fi
  echo -e "${GREEN}✓ Stopped.${NC}"
  echo -e "  To stop Ollama: docker stop $OLLAMA_CONTAINER"
  echo -e "  To remove venv: rm -rf $VENV_DIR"
}

trap cleanup EXIT INT TERM

# Keep running
wait
