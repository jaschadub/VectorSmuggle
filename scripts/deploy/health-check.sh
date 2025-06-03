#!/bin/bash

# VectorSmuggle Health Check Script
# Performs comprehensive health checks for deployed services

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PLATFORM="${PLATFORM:-docker-compose}"
NAMESPACE="${NAMESPACE:-vectorsmuggle}"
TIMEOUT="${TIMEOUT:-300}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Help function
show_help() {
    cat << EOF
VectorSmuggle Health Check Script

Usage: $0 [OPTIONS]

Options:
    -h, --help                    Show this help message
    -p, --platform PLATFORM      Set platform (docker-compose|kubernetes) (default: docker-compose)
    -n, --namespace NAMESPACE     Set Kubernetes namespace (default: vectorsmuggle)
    -t, --timeout TIMEOUT        Set timeout in seconds (default: 300)
    --quick                      Run quick health checks only
    --detailed                   Run detailed health checks
    --continuous                 Run continuous monitoring
    --export FILE                Export results to file

Examples:
    $0                                    # Basic health check
    $0 --platform kubernetes --namespace vectorsmuggle  # Kubernetes health check
    $0 --detailed --export health-report.json           # Detailed check with export

EOF
}

# Parse command line arguments
QUICK=false
DETAILED=false
CONTINUOUS=false
EXPORT_FILE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -p|--platform)
            PLATFORM="$2"
            shift 2
            ;;
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        -t|--timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --quick)
            QUICK=true
            shift
            ;;
        --detailed)
            DETAILED=true
            shift
            ;;
        --continuous)
            CONTINUOUS=true
            shift
            ;;
        --export)
            EXPORT_FILE="$2"
            shift 2
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Health check results
HEALTH_RESULTS=()

# Add result to array
add_result() {
    local service="$1"
    local status="$2"
    local message="$3"
    local timestamp="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
    
    HEALTH_RESULTS+=("{\"service\":\"$service\",\"status\":\"$status\",\"message\":\"$message\",\"timestamp\":\"$timestamp\"}")
}

# Wait for service to be ready
wait_for_service() {
    local service="$1"
    local check_command="$2"
    local max_attempts=$((TIMEOUT / 10))
    local attempt=1
    
    log_info "Waiting for $service to be ready..."
    
    while [[ $attempt -le $max_attempts ]]; do
        if eval "$check_command" &> /dev/null; then
            log_success "$service is ready"
            add_result "$service" "healthy" "Service is responding"
            return 0
        fi
        
        log_info "Attempt $attempt/$max_attempts: $service not ready yet..."
        sleep 10
        ((attempt++))
    done
    
    log_error "$service failed to become ready within $TIMEOUT seconds"
    add_result "$service" "unhealthy" "Service failed to become ready within timeout"
    return 1
}

# Check Docker Compose services
check_docker_compose_health() {
    log_info "Checking Docker Compose services health..."
    
    cd "$PROJECT_ROOT"
    
    # Determine compose command
    if command -v docker-compose &> /dev/null; then
        COMPOSE_CMD="docker-compose"
    else
        COMPOSE_CMD="docker compose"
    fi
    
    # Check if services are running
    if ! $COMPOSE_CMD ps --services --filter "status=running" | grep -q .; then
        log_error "No running services found"
        add_result "docker-compose" "unhealthy" "No running services found"
        return 1
    fi
    
    # Check VectorSmuggle service
    if $COMPOSE_CMD ps vectorsmuggle | grep -q "Up"; then
        # Test configuration loading
        wait_for_service "vectorsmuggle" "$COMPOSE_CMD exec -T vectorsmuggle python -c 'from config import get_config; get_config()'"
        
        # Test basic functionality if detailed check
        if [[ "$DETAILED" == true ]]; then
            log_info "Running detailed VectorSmuggle checks..."
            
            # Check if scripts are accessible
            if $COMPOSE_CMD exec -T vectorsmuggle test -f /app/scripts/embed.py; then
                add_result "vectorsmuggle-scripts" "healthy" "Scripts are accessible"
            else
                add_result "vectorsmuggle-scripts" "unhealthy" "Scripts not found"
            fi
            
            # Check if modules can be imported
            if $COMPOSE_CMD exec -T vectorsmuggle python -c "import steganography, evasion, query, loaders" &> /dev/null; then
                add_result "vectorsmuggle-modules" "healthy" "All modules can be imported"
            else
                add_result "vectorsmuggle-modules" "unhealthy" "Module import failed"
            fi
        fi
    else
        log_error "VectorSmuggle service is not running"
        add_result "vectorsmuggle" "unhealthy" "Service is not running"
    fi
    
    # Check Qdrant service if running
    if $COMPOSE_CMD ps qdrant | grep -q "Up"; then
        wait_for_service "qdrant" "curl -f http://localhost:6333/health"
        
        if [[ "$DETAILED" == true ]]; then
            # Check Qdrant collections
            if curl -s http://localhost:6333/collections | grep -q "collections"; then
                add_result "qdrant-api" "healthy" "API is responding"
            else
                add_result "qdrant-api" "unhealthy" "API not responding properly"
            fi
        fi
    fi
    
    # Check Redis service if running
    if $COMPOSE_CMD ps redis | grep -q "Up"; then
        wait_for_service "redis" "docker exec vectorsmuggle-redis redis-cli ping"
    fi
}

# Check Kubernetes services
check_kubernetes_health() {
    log_info "Checking Kubernetes services health..."
    
    # Check if namespace exists
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_error "Namespace $NAMESPACE does not exist"
        add_result "kubernetes-namespace" "unhealthy" "Namespace does not exist"
        return 1
    fi
    
    # Check deployments
    log_info "Checking deployments..."
    
    # VectorSmuggle deployment
    if kubectl get deployment vectorsmuggle -n "$NAMESPACE" &> /dev/null; then
        if kubectl rollout status deployment/vectorsmuggle -n "$NAMESPACE" --timeout=60s &> /dev/null; then
            add_result "vectorsmuggle-deployment" "healthy" "Deployment is ready"
            
            # Check pod health
            local ready_pods=$(kubectl get pods -n "$NAMESPACE" -l app=vectorsmuggle --field-selector=status.phase=Running --no-headers | wc -l)
            local total_pods=$(kubectl get pods -n "$NAMESPACE" -l app=vectorsmuggle --no-headers | wc -l)
            
            if [[ "$ready_pods" -eq "$total_pods" ]] && [[ "$ready_pods" -gt 0 ]]; then
                add_result "vectorsmuggle-pods" "healthy" "$ready_pods/$total_pods pods ready"
            else
                add_result "vectorsmuggle-pods" "unhealthy" "$ready_pods/$total_pods pods ready"
            fi
        else
            add_result "vectorsmuggle-deployment" "unhealthy" "Deployment not ready"
        fi
    else
        add_result "vectorsmuggle-deployment" "unhealthy" "Deployment not found"
    fi
    
    # Qdrant deployment
    if kubectl get deployment qdrant -n "$NAMESPACE" &> /dev/null; then
        if kubectl rollout status deployment/qdrant -n "$NAMESPACE" --timeout=60s &> /dev/null; then
            add_result "qdrant-deployment" "healthy" "Deployment is ready"
        else
            add_result "qdrant-deployment" "unhealthy" "Deployment not ready"
        fi
    fi
    
    # Check services
    log_info "Checking services..."
    
    if kubectl get service vectorsmuggle -n "$NAMESPACE" &> /dev/null; then
        local endpoints=$(kubectl get endpoints vectorsmuggle -n "$NAMESPACE" -o jsonpath='{.subsets[*].addresses[*].ip}' | wc -w)
        if [[ "$endpoints" -gt 0 ]]; then
            add_result "vectorsmuggle-service" "healthy" "$endpoints endpoints available"
        else
            add_result "vectorsmuggle-service" "unhealthy" "No endpoints available"
        fi
    else
        add_result "vectorsmuggle-service" "unhealthy" "Service not found"
    fi
    
    # Detailed checks
    if [[ "$DETAILED" == true ]]; then
        log_info "Running detailed Kubernetes checks..."
        
        # Check resource usage
        local cpu_usage=$(kubectl top pods -n "$NAMESPACE" --no-headers 2>/dev/null | awk '{sum+=$2} END {print sum}' || echo "0")
        local memory_usage=$(kubectl top pods -n "$NAMESPACE" --no-headers 2>/dev/null | awk '{sum+=$3} END {print sum}' || echo "0")
        
        add_result "resource-usage" "info" "CPU: ${cpu_usage}m, Memory: ${memory_usage}Mi"
        
        # Check persistent volumes
        local pvcs=$(kubectl get pvc -n "$NAMESPACE" --no-headers | wc -l)
        add_result "persistent-volumes" "info" "$pvcs PVCs found"
        
        # Check ingress
        if kubectl get ingress -n "$NAMESPACE" &> /dev/null; then
            add_result "ingress" "healthy" "Ingress configured"
        else
            add_result "ingress" "warning" "No ingress found"
        fi
    fi
}

# Export results
export_results() {
    if [[ -z "$EXPORT_FILE" ]]; then
        return 0
    fi
    
    log_info "Exporting results to $EXPORT_FILE..."
    
    local json_output="{"
    json_output+="\"timestamp\":\"$(date -u +'%Y-%m-%dT%H:%M:%SZ')\","
    json_output+="\"platform\":\"$PLATFORM\","
    json_output+="\"namespace\":\"$NAMESPACE\","
    json_output+="\"results\":["
    
    local first=true
    for result in "${HEALTH_RESULTS[@]}"; do
        if [[ "$first" == true ]]; then
            first=false
        else
            json_output+=","
        fi
        json_output+="$result"
    done
    
    json_output+="]}"
    
    echo "$json_output" | python -m json.tool > "$EXPORT_FILE"
    log_success "Results exported to $EXPORT_FILE"
}

# Print summary
print_summary() {
    echo
    log_info "Health Check Summary:"
    echo "===================="
    
    local healthy=0
    local unhealthy=0
    local warnings=0
    
    for result in "${HEALTH_RESULTS[@]}"; do
        local service=$(echo "$result" | python -c "import sys, json; print(json.load(sys.stdin)['service'])")
        local status=$(echo "$result" | python -c "import sys, json; print(json.load(sys.stdin)['status'])")
        local message=$(echo "$result" | python -c "import sys, json; print(json.load(sys.stdin)['message'])")
        
        case $status in
            healthy)
                echo -e "${GREEN}✓${NC} $service: $message"
                ((healthy++))
                ;;
            unhealthy)
                echo -e "${RED}✗${NC} $service: $message"
                ((unhealthy++))
                ;;
            warning)
                echo -e "${YELLOW}⚠${NC} $service: $message"
                ((warnings++))
                ;;
            info)
                echo -e "${BLUE}ℹ${NC} $service: $message"
                ;;
        esac
    done
    
    echo
    log_info "Total: $((healthy + unhealthy + warnings)) checks"
    log_success "Healthy: $healthy"
    if [[ $warnings -gt 0 ]]; then
        log_warning "Warnings: $warnings"
    fi
    if [[ $unhealthy -gt 0 ]]; then
        log_error "Unhealthy: $unhealthy"
    fi
    
    if [[ $unhealthy -gt 0 ]]; then
        return 1
    else
        return 0
    fi
}

# Continuous monitoring
run_continuous_monitoring() {
    log_info "Starting continuous monitoring (Ctrl+C to stop)..."
    
    while true; do
        clear
        echo "VectorSmuggle Health Monitor - $(date)"
        echo "================================"
        
        HEALTH_RESULTS=()
        
        if [[ "$PLATFORM" == "docker-compose" ]]; then
            check_docker_compose_health
        elif [[ "$PLATFORM" == "kubernetes" ]]; then
            check_kubernetes_health
        fi
        
        print_summary
        
        sleep 30
    done
}

# Main execution
main() {
    log_info "Starting VectorSmuggle health check..."
    log_info "Platform: $PLATFORM"
    
    if [[ "$PLATFORM" == "kubernetes" ]]; then
        log_info "Namespace: $NAMESPACE"
    fi
    
    if [[ "$CONTINUOUS" == true ]]; then
        run_continuous_monitoring
        return $?
    fi
    
    if [[ "$PLATFORM" == "docker-compose" ]]; then
        check_docker_compose_health
    elif [[ "$PLATFORM" == "kubernetes" ]]; then
        check_kubernetes_health
    else
        log_error "Unsupported platform: $PLATFORM"
        exit 1
    fi
    
    export_results
    print_summary
    
    local exit_code=$?
    
    if [[ $exit_code -eq 0 ]]; then
        log_success "All health checks passed!"
    else
        log_error "Some health checks failed!"
    fi
    
    return $exit_code
}

# Handle Ctrl+C gracefully
trap 'log_info "Health check interrupted"; exit 0' INT

# Run main function
main "$@"