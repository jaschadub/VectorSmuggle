#!/bin/bash

# VectorSmuggle Cleanup Script
# Cleans up Docker containers, images, volumes, and Kubernetes resources

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PLATFORM="${PLATFORM:-docker-compose}"
NAMESPACE="${NAMESPACE:-vectorsmuggle}"

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
VectorSmuggle Cleanup Script

Usage: $0 [OPTIONS]

Options:
    -h, --help                    Show this help message
    -p, --platform PLATFORM      Set platform (docker-compose|kubernetes|all) (default: docker-compose)
    -n, --namespace NAMESPACE     Set Kubernetes namespace (default: vectorsmuggle)
    --containers                 Clean up containers only
    --images                     Clean up images only
    --volumes                    Clean up volumes only
    --networks                   Clean up networks only
    --all                        Clean up everything (containers, images, volumes, networks)
    --force                      Force cleanup without confirmation
    --dry-run                    Show what would be cleaned without executing

Platforms:
    docker-compose              Clean up Docker Compose resources
    kubernetes                  Clean up Kubernetes resources
    all                         Clean up both Docker and Kubernetes resources

Examples:
    $0                                    # Clean up Docker Compose deployment
    $0 --platform kubernetes             # Clean up Kubernetes deployment
    $0 --all --force                     # Force clean everything
    $0 --dry-run --all                   # Show what would be cleaned

EOF
}

# Parse command line arguments
CONTAINERS=false
IMAGES=false
VOLUMES=false
NETWORKS=false
ALL=false
FORCE=false
DRY_RUN=false

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
        --containers)
            CONTAINERS=true
            shift
            ;;
        --images)
            IMAGES=true
            shift
            ;;
        --volumes)
            VOLUMES=true
            shift
            ;;
        --networks)
            NETWORKS=true
            shift
            ;;
        --all)
            ALL=true
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Set defaults if no specific cleanup type specified
if [[ "$CONTAINERS" == false ]] && [[ "$IMAGES" == false ]] && [[ "$VOLUMES" == false ]] && [[ "$NETWORKS" == false ]] && [[ "$ALL" == false ]]; then
    CONTAINERS=true
    VOLUMES=true
fi

# If --all is specified, enable everything
if [[ "$ALL" == true ]]; then
    CONTAINERS=true
    IMAGES=true
    VOLUMES=true
    NETWORKS=true
fi

# Confirmation prompt
confirm_cleanup() {
    if [[ "$FORCE" == true ]] || [[ "$DRY_RUN" == true ]]; then
        return 0
    fi
    
    echo
    log_warning "This will clean up VectorSmuggle resources:"
    [[ "$CONTAINERS" == true ]] && echo "  - Containers"
    [[ "$IMAGES" == true ]] && echo "  - Images"
    [[ "$VOLUMES" == true ]] && echo "  - Volumes"
    [[ "$NETWORKS" == true ]] && echo "  - Networks"
    echo "  - Platform: $PLATFORM"
    [[ "$PLATFORM" == "kubernetes" ]] && echo "  - Namespace: $NAMESPACE"
    echo
    
    read -p "Are you sure you want to continue? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Cleanup cancelled"
        exit 0
    fi
}

# Execute command with dry-run support
execute_command() {
    local cmd="$1"
    local description="$2"
    
    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY-RUN] $description"
        echo "  Command: $cmd"
    else
        log_info "$description"
        if eval "$cmd"; then
            log_success "$description completed"
        else
            log_warning "$description failed or had no effect"
        fi
    fi
}

# Clean up Docker Compose resources
cleanup_docker_compose() {
    log_info "Cleaning up Docker Compose resources..."
    
    cd "$PROJECT_ROOT"
    
    # Determine compose command
    if command -v docker-compose &> /dev/null; then
        COMPOSE_CMD="docker-compose"
    else
        COMPOSE_CMD="docker compose"
    fi
    
    # Determine compose files
    COMPOSE_FILES=("-f" "docker-compose.yml")
    if [[ -f "docker-compose.dev.yml" ]]; then
        COMPOSE_FILES+=("-f" "docker-compose.dev.yml")
    fi
    if [[ -f "docker-compose.prod.yml" ]]; then
        COMPOSE_FILES+=("-f" "docker-compose.prod.yml")
    fi
    
    # Stop and remove containers
    if [[ "$CONTAINERS" == true ]]; then
        execute_command "$COMPOSE_CMD ${COMPOSE_FILES[*]} down --remove-orphans" "Stopping and removing containers"
    fi
    
    # Remove images
    if [[ "$IMAGES" == true ]]; then
        execute_command "$COMPOSE_CMD ${COMPOSE_FILES[*]} down --rmi all" "Removing images"
        
        # Remove VectorSmuggle images specifically
        local vectorsmuggle_images=$(docker images --format "{{.Repository}}:{{.Tag}}" | grep vectorsmuggle || true)
        if [[ -n "$vectorsmuggle_images" ]]; then
            execute_command "docker rmi $vectorsmuggle_images" "Removing VectorSmuggle images"
        fi
    fi
    
    # Remove volumes
    if [[ "$VOLUMES" == true ]]; then
        execute_command "$COMPOSE_CMD ${COMPOSE_FILES[*]} down --volumes" "Removing volumes"
        
        # Remove named volumes specifically
        local vectorsmuggle_volumes=$(docker volume ls --format "{{.Name}}" | grep vectorsmuggle || true)
        if [[ -n "$vectorsmuggle_volumes" ]]; then
            execute_command "docker volume rm $vectorsmuggle_volumes" "Removing VectorSmuggle volumes"
        fi
    fi
    
    # Remove networks
    if [[ "$NETWORKS" == true ]]; then
        local vectorsmuggle_networks=$(docker network ls --format "{{.Name}}" | grep vectorsmuggle || true)
        if [[ -n "$vectorsmuggle_networks" ]]; then
            execute_command "docker network rm $vectorsmuggle_networks" "Removing VectorSmuggle networks"
        fi
    fi
    
    # Clean up dangling resources
    if [[ "$ALL" == true ]]; then
        execute_command "docker system prune -f" "Cleaning up dangling resources"
    fi
}

# Clean up Kubernetes resources
cleanup_kubernetes() {
    log_info "Cleaning up Kubernetes resources..."
    
    # Check if namespace exists
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_warning "Namespace $NAMESPACE does not exist"
        return 0
    fi
    
    cd "$PROJECT_ROOT"
    
    # Remove deployments
    if [[ "$CONTAINERS" == true ]]; then
        execute_command "kubectl delete deployment --all -n $NAMESPACE" "Removing deployments"
        execute_command "kubectl delete pod --all -n $NAMESPACE" "Removing pods"
    fi
    
    # Remove services
    execute_command "kubectl delete service --all -n $NAMESPACE" "Removing services"
    
    # Remove ingress
    execute_command "kubectl delete ingress --all -n $NAMESPACE" "Removing ingress"
    
    # Remove configmaps and secrets
    execute_command "kubectl delete configmap --all -n $NAMESPACE" "Removing configmaps"
    execute_command "kubectl delete secret --all -n $NAMESPACE" "Removing secrets"
    
    # Remove persistent volume claims
    if [[ "$VOLUMES" == true ]]; then
        execute_command "kubectl delete pvc --all -n $NAMESPACE" "Removing persistent volume claims"
    fi
    
    # Remove network policies
    if [[ "$NETWORKS" == true ]]; then
        execute_command "kubectl delete networkpolicy --all -n $NAMESPACE" "Removing network policies"
    fi
    
    # Remove namespace if cleaning everything
    if [[ "$ALL" == true ]]; then
        execute_command "kubectl delete namespace $NAMESPACE" "Removing namespace"
    fi
}

# Clean up local development files
cleanup_local_files() {
    log_info "Cleaning up local development files..."
    
    cd "$PROJECT_ROOT"
    
    # Remove cache directories
    local cache_dirs=(".query_cache" "faiss_index" "__pycache__" ".pytest_cache" ".ruff_cache")
    for dir in "${cache_dirs[@]}"; do
        if [[ -d "$dir" ]]; then
            execute_command "rm -rf $dir" "Removing $dir"
        fi
    done
    
    # Remove log files
    if [[ -d "logs" ]]; then
        execute_command "rm -rf logs/*" "Removing log files"
    fi
    
    # Remove temporary files
    execute_command "find . -name '*.pyc' -delete" "Removing Python cache files"
    execute_command "find . -name '*.pyo' -delete" "Removing Python optimized files"
    execute_command "find . -name '*~' -delete" "Removing backup files"
    
    # Remove build artifacts
    if [[ -f "bandit-report.json" ]]; then
        execute_command "rm bandit-report.json" "Removing bandit report"
    fi
}

# Show cleanup summary
show_cleanup_summary() {
    echo
    log_info "Cleanup Summary:"
    echo "================"
    
    if [[ "$PLATFORM" == "docker-compose" ]] || [[ "$PLATFORM" == "all" ]]; then
        echo "Docker Compose:"
        [[ "$CONTAINERS" == true ]] && echo "  ✓ Containers stopped and removed"
        [[ "$IMAGES" == true ]] && echo "  ✓ Images removed"
        [[ "$VOLUMES" == true ]] && echo "  ✓ Volumes removed"
        [[ "$NETWORKS" == true ]] && echo "  ✓ Networks removed"
    fi
    
    if [[ "$PLATFORM" == "kubernetes" ]] || [[ "$PLATFORM" == "all" ]]; then
        echo "Kubernetes (namespace: $NAMESPACE):"
        [[ "$CONTAINERS" == true ]] && echo "  ✓ Deployments and pods removed"
        echo "  ✓ Services and ingress removed"
        echo "  ✓ ConfigMaps and secrets removed"
        [[ "$VOLUMES" == true ]] && echo "  ✓ Persistent volume claims removed"
        [[ "$NETWORKS" == true ]] && echo "  ✓ Network policies removed"
        [[ "$ALL" == true ]] && echo "  ✓ Namespace removed"
    fi
    
    echo "Local files:"
    echo "  ✓ Cache directories cleaned"
    echo "  ✓ Log files removed"
    echo "  ✓ Temporary files cleaned"
    
    if [[ "$DRY_RUN" == true ]]; then
        log_info "This was a dry run. No actual cleanup was performed."
    else
        log_success "Cleanup completed successfully!"
    fi
}

# Main execution
main() {
    log_info "Starting VectorSmuggle cleanup..."
    log_info "Platform: $PLATFORM"
    
    if [[ "$PLATFORM" == "kubernetes" ]]; then
        log_info "Namespace: $NAMESPACE"
    fi
    
    confirm_cleanup
    
    case $PLATFORM in
        docker-compose)
            cleanup_docker_compose
            ;;
        kubernetes)
            cleanup_kubernetes
            ;;
        all)
            cleanup_docker_compose
            cleanup_kubernetes
            ;;
        *)
            log_error "Invalid platform: $PLATFORM"
            log_error "Valid platforms: docker-compose, kubernetes, all"
            exit 1
            ;;
    esac
    
    cleanup_local_files
    show_cleanup_summary
}

# Run main function
main "$@"