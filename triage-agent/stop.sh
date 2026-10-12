#!/usr/bin/env bash
# ==============================================================================
#  OmniSpec-Triage Process Terminator Script
#  Gracefully stops running web servers, voice pipelines, and edge resources
# ==============================================================================

# Terminal styling
BOLD='\033[1m'
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

PORT=8000

print_banner() {
    echo -e "${CYAN}${BOLD}"
    echo "================================================================================"
    echo "  🛑  OMNISPEC-TRIAGE : Service Shutdown & Resource Cleaner"
    echo "================================================================================"
    echo -e "${NC}"
}

# Find all relevant PIDs
get_triage_pids() {
    pgrep -f "run_web\.py|src\.pipeline|src\.pipeline_speculative|server:app|benchmark_latency" || true
}

# Find process bound to port 8000
get_port_pids() {
    if command -v fuser >/dev/null 2>&1; then
        fuser 8000/tcp 2>/dev/null || true
    elif command -v lsof >/dev/null 2>&1; then
        lsof -ti:8000 2>/dev/null || true
    elif command -v ss >/dev/null 2>&1; then
        ss -lptn 'sport = :8000' 2>/dev/null | grep -oP 'pid=\K[0-9]+' || true
    fi
}

# Show running triage status
show_status() {
    print_banner
    echo -e "${BOLD}🔍 Current Process & Resource Status:${NC}\n"

    # Check triage python processes
    PIDS=$(get_triage_pids)
    if [ -n "$PIDS" ]; then
        echo -e "  • ${YELLOW}Active Triage Processes:${NC}"
        ps -o pid,user,%cpu,%mem,command -p $PIDS | sed 's/^/    /'
    else
        echo -e "  • Triage Processes:  ${GREEN}✓ None running${NC}"
    fi

    # Check port 8000
    PORT_PID=$(get_port_pids)
    if [ -n "$PORT_PID" ]; then
        echo -e "  • Port 8000 Status:   ${YELLOW}⚠️ In use by PID: $PORT_PID${NC}"
    else
        echo -e "  • Port 8000 Status:   ${GREEN}✓ Free${NC}"
    fi

    # Check Ollama VRAM usage
    if command -v ollama >/dev/null 2>&1 && curl -s http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
        ACTIVE_MODELS=$(ollama ps 2>/dev/null | awk 'NR>1 {print $1}')
        if [ -n "$ACTIVE_MODELS" ]; then
            echo -e "  • Ollama GPU VRAM:   ${YELLOW}Resident model(s): ${ACTIVE_MODELS}${NC}"
        else
            echo -e "  • Ollama GPU VRAM:   ${GREEN}✓ 0 MB (No models resident)${NC}"
        fi
    fi
    echo ""
}

# Unload models from Ollama VRAM
unload_vram() {
    if command -v ollama >/dev/null 2>&1 && curl -s http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
        MODELS=$(ollama ps 2>/dev/null | awk 'NR>1 {print $1}')
        if [ -n "$MODELS" ]; then
            echo -e "${CYAN}🧹 Unloading active Ollama models to free GPU VRAM...${NC}"
            for m in $MODELS; do
                ollama stop "$m" >/dev/null 2>&1 || true
                echo -e "  ${GREEN}✓ Unloaded model: $m${NC}"
            done
        else
            echo -e "${GREEN}✓ No models currently resident in VRAM.${NC}"
        fi
    fi
}

# Stop all triage services
stop_services() {
    local include_vram=$1
    print_banner

    PIDS=$(get_triage_pids)
    PORT_PIDS=$(get_port_pids)
    ALL_PIDS=$(echo "$PIDS $PORT_PIDS" | tr ' ' '\n' | grep -v '^$' | sort -u || true)

    if [ -z "$ALL_PIDS" ]; then
        echo -e "${GREEN}✓ No OmniSpec-Triage processes are currently running.${NC}"
    else
        echo -e "${YELLOW}Halting running processes:${NC}"
        for pid in $ALL_PIDS; do
            CMD=$(ps -p "$pid" -o comm= 2>/dev/null || echo "process")
            echo -e "  Stopping PID ${BOLD}$pid${NC} ($CMD)..."
            kill -15 "$pid" 2>/dev/null || true
        done

        # Wait briefly for graceful shutdown
        sleep 1.5

        # Check for remaining processes and force terminate if needed
        REMAINING=""
        for pid in $ALL_PIDS; do
            if kill -0 "$pid" 2>/dev/null; then
                REMAINING="$REMAINING $pid"
            fi
        done

        if [ -n "$REMAINING" ]; then
            echo -e "${RED}⚠️ Forcing termination for unresponsive processes...${NC}"
            for pid in $REMAINING; do
                kill -9 "$pid" 2>/dev/null || true
                echo -e "  Force killed PID $pid"
            done
        fi

        echo -e "${GREEN}✓ All OmniSpec-Triage processes stopped successfully.${NC}"
    fi

    # Unload VRAM if requested
    if [ "$include_vram" = "true" ]; then
        echo ""
        unload_vram
    fi

    echo -e "\n${BOLD}Final check:${NC}"
    PORT_CHECK=$(get_port_pids)
    if [ -z "$PORT_CHECK" ]; then
        echo -e "  • Port 8000: ${GREEN}✓ Released and ready${NC}"
    else
        echo -e "  • Port 8000: ${RED}✗ Still occupied by PID $PORT_CHECK${NC}"
    fi
    echo ""
}

# Command dispatch
case "$1" in
    status|--status|-s)
        show_status
        ;;
    vram|--vram)
        print_banner
        unload_vram
        ;;
    all|--all)
        stop_services true
        ;;
    help|--help|-h)
        print_banner
        echo -e "${BOLD}Usage:${NC} ./stop.sh [options]"
        echo ""
        echo "Options:"
        echo "  (no args)           Stop all triage web servers and live pipelines"
        echo "  all | --all         Stop all triage processes AND free Ollama GPU VRAM"
        echo "  vram | --vram       Unload models from GPU VRAM only (leave services)"
        echo "  status | -s         Show current running processes, port 8000, and VRAM"
        echo "  help | -h           Show this help message"
        echo ""
        ;;
    *)
        stop_services false
        ;;
esac
