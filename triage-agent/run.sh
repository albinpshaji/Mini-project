#!/usr/bin/env bash
# ==============================================================================
#  OmniSpec-Triage Unified Execution Script
#  Edge-Native Speculative Multimodal Hospital Intake & Triage Agent
# ==============================================================================

set -e

# Terminal styling
BOLD='\033[1m'
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Determine script root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Locate Python virtual environment
if [ -f "$SCRIPT_DIR/venv/bin/python" ]; then
    PYTHON_BIN="$SCRIPT_DIR/venv/bin/python"
elif [ -n "$VIRTUAL_ENV" ] && [ -f "$VIRTUAL_ENV/bin/python" ]; then
    PYTHON_BIN="$VIRTUAL_ENV/bin/python"
else
    PYTHON_BIN="$(command -v python3 || command -v python)"
fi

# Print project banner
print_banner() {
    echo -e "${CYAN}${BOLD}"
    echo "================================================================================"
    echo "  🏥  OMNISPEC-TRIAGE : Edge Multimodal Hospital Intake Agent"
    echo "================================================================================"
    echo -e "${NC}"
}

# Perform preflight diagnostics & environment status checks
check_environment() {
    echo -e "${BOLD}🔍 Environment Preflight Check:${NC}"

    # 1. Python binary
    echo -e "  • Python Engine:     ${GREEN}✓ ${PYTHON_BIN}${NC}"

    # 2. Ollama & LLM status
    if curl -s http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
        if curl -s http://127.0.0.1:11434/api/tags | grep -q "qwen3.5:4b"; then
            echo -e "  • Ollama Service:    ${GREEN}✓ Active (Loaded: qwen3.5:4b)${NC}"
        elif curl -s http://127.0.0.1:11434/api/tags | grep -q "qwen2.5:1.5b"; then
            echo -e "  • Ollama Service:    ${GREEN}✓ Active (Loaded: qwen2.5:1.5b)${NC}"
        else
            echo -e "  • Ollama Service:    ${YELLOW}⚠️ Active, but default triage models not found (Run: ollama pull qwen3.5:4b)${NC}"
        fi
    else
        echo -e "  • Ollama Service:    ${RED}✗ Not running! (Start with: ollama serve)${NC}"
    fi

    # 3. Piper TTS voice model
    if [ -f "$SCRIPT_DIR/models/tts/en_US-lessac-medium.onnx" ]; then
        echo -e "  • Piper TTS Voice:   ${GREEN}✓ Found (en_US-lessac-medium.onnx)${NC}"
    else
        echo -e "  • Piper TTS Voice:   ${RED}✗ Missing from models/tts/!${NC}"
    fi

    # 4. Frontend bundle
    if [ -f "$SCRIPT_DIR/frontend/dist/index.html" ]; then
        echo -e "  • Web Kiosk Build:   ${GREEN}✓ Ready (frontend/dist)${NC}"
    else
        echo -e "  • Web Kiosk Build:   ${YELLOW}⚠️ Needs build (frontend/dist missing)${NC}"
    fi

    # 5. Edge GPU detection
    if command -v nvidia-smi >/dev/null 2>&1; then
        GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -n 1 || true)
        if [ -n "$GPU_INFO" ]; then
            echo -e "  • Edge GPU Target:   ${GREEN}✓ ${GPU_INFO}${NC}"
        else
            echo -e "  • Edge GPU Target:   ${YELLOW}CPU-only mode${NC}"
        fi
    else
        echo -e "  • Edge GPU Target:   ${YELLOW}CPU-only mode (nvidia-smi not in PATH)${NC}"
    fi
    echo ""
}

# Run Web & Mobile Interface (HTTP)
run_web() {
    print_banner
    ensure_frontend_built
    echo -e "${GREEN}${BOLD}🚀 Launching OmniSpec-Triage Web & Mobile Server (HTTP)...${NC}"
    exec "$PYTHON_BIN" run_web.py --port 8000 "$@"
}

# Run Web & Mobile Interface with self-signed SSL (HTTPS)
run_web_ssl() {
    print_banner
    ensure_frontend_built
    echo -e "${GREEN}${BOLD}🔐 Launching OmniSpec-Triage Web & Mobile Server with SSL (HTTPS)...${NC}"
    echo -e "${CYAN}ℹ️  Use this mode for direct mobile phone microphone streaming over local Wi-Fi.${NC}"
    exec "$PYTHON_BIN" run_web.py --ssl --port 8000 "$@"
}

# Run Speculative Live Audio Pipeline (CLI Mic)
run_speculative() {
    print_banner
    echo -e "${PURPLE}${BOLD}⚡ Launching Speculative Voice Pipeline (Real-Time Audio)...${NC}"
    echo -e "${CYAN}ℹ️  Speak into your microphone. Speculative SLM reasoning will run concurrently.${NC}"
    exec "$PYTHON_BIN" -m src.pipeline_speculative "$@"
}

# Run Sequential Baseline Pipeline (CLI Mic)
run_sequential() {
    print_banner
    echo -e "${BLUE}${BOLD}🐢 Launching Sequential Baseline Voice Pipeline (Real-Time Audio)...${NC}"
    exec "$PYTHON_BIN" -m src.pipeline "$@"
}

# Run Latency Benchmark (Evaluation)
run_benchmark() {
    print_banner
    echo -e "${YELLOW}${BOLD}📊 Running Sequential vs. Speculative Latency Benchmark Suite...${NC}"
    exec "$PYTHON_BIN" -m eval.benchmark_latency "$@"
}

# Run Component Diagnostics Suite
run_diagnostics() {
    print_banner
    echo -e "${CYAN}${BOLD}🏥 Running Component Diagnostics & Unit Verification...${NC}"
    "$PYTHON_BIN" tests/run_diagnostics.py
    echo ""
    echo -e "${CYAN}${BOLD}🏥 Running Hospital Router Tests...${NC}"
    "$PYTHON_BIN" -m tests.test_hospital_router
}

# Ensure Frontend is built
ensure_frontend_built() {
    if [ ! -f "$SCRIPT_DIR/frontend/dist/index.html" ]; then
        echo -e "${YELLOW}📦 Frontend dist not found. Building now...${NC}"
        build_frontend
    fi
}

# Build/Rebuild React Frontend
build_frontend() {
    echo -e "${CYAN}${BOLD}🔨 Building React frontend with Vite...${NC}"
    if command -v npm >/dev/null 2>&1; then
        (cd "$SCRIPT_DIR/frontend" && npm install && npm run build)
        echo -e "${GREEN}✓ Frontend build complete!${NC}\n"
    else
        echo -e "${RED}✗ Error: npm is not installed. Cannot build frontend.${NC}"
        exit 1
    fi
}

# Show interactive menu
show_menu() {
    print_banner
    check_environment

    echo -e "${BOLD}Select execution mode:${NC}"
    echo -e "  ${GREEN}${BOLD}[1]${NC} Web & Mobile Interface ${BOLD}(Recommended)${NC}   - Browser GUI on port 8000"
    echo -e "  ${GREEN}${BOLD}[2]${NC} Web & Mobile Interface with HTTPS/SSL     - Enables remote smartphone mic"
    echo -e "  ${PURPLE}${BOLD}[3]${NC} Live Mic Speculative Pipeline (CLI)        - Full-Duplex AEC + Speculative SLM"
    echo -e "  ${BLUE}${BOLD}[4]${NC} Live Mic Sequential Baseline Pipeline (CLI) - Standard serialized cascade"
    echo -e "  ${YELLOW}${BOLD}[5]${NC} Latency & Speedup Benchmark                - Test canned clinical cases"
    echo -e "  ${CYAN}${BOLD}[6]${NC} System Diagnostics & Health Suite          - Test Audio, Safety, SLM, TTS"
    echo -e "  ${BOLD}[7]${NC} Rebuild React Frontend Dashboard           - npm run build"
    echo -e "  ${RED}${BOLD}[q]${NC} Quit"
    echo ""
    read -rp "Enter choice [1-7, q]: " choice

    case "$choice" in
        1) run_web ;;
        2) run_web_ssl ;;
        3) run_speculative ;;
        4) run_sequential ;;
        5) run_benchmark ;;
        6) run_diagnostics ;;
        7) build_frontend ;;
        q|Q) echo -e "${CYAN}Exiting. Stay safe!${NC}"; exit 0 ;;
        *) echo -e "${RED}Invalid selection.${NC}"; exit 1 ;;
    esac
}

# CLI Argument parsing
if [ $# -eq 0 ]; then
    show_menu
else
    case "$1" in
        web|--web)
            shift
            run_web "$@"
            ;;
        ssl|--ssl)
            shift
            run_web_ssl "$@"
            ;;
        spec|speculative|--spec|--speculative)
            shift
            run_speculative "$@"
            ;;
        seq|sequential|--seq|--sequential)
            shift
            run_sequential "$@"
            ;;
        bench|benchmark|eval|--benchmark|--eval)
            shift
            run_benchmark "$@"
            ;;
        diag|diagnostic|test|--diag|--test)
            shift
            run_diagnostics "$@"
            ;;
        build|--build)
            shift
            build_frontend
            ;;
        check|--check)
            print_banner
            check_environment
            ;;
        help|-h|--help)
            print_banner
            echo -e "${BOLD}Usage:${NC} ./run.sh [command] [options]"
            echo ""
            echo "Commands:"
            echo "  (no args)           Open interactive selection menu"
            echo "  web                 Run Web & Mobile interface (HTTP)"
            echo "  ssl                 Run Web & Mobile interface with SSL (HTTPS)"
            echo "  speculative | spec  Run Live Mic Speculative pipeline"
            echo "  sequential | seq    Run Live Mic Sequential baseline pipeline"
            echo "  benchmark | eval    Run Latency Benchmark (no mic needed)"
            echo "  test | diag         Run System Diagnostics suite"
            echo "  build               Rebuild the React frontend"
            echo "  check               Display environment and dependency status"
            echo "  help | -h           Show this help message"
            echo ""
            ;;
        *)
            echo -e "${RED}Unknown command: $1${NC}"
            echo "Run './run.sh help' for usage instructions."
            exit 1
            ;;
    esac
fi
