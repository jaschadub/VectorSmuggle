#!/bin/bash

# VectorSmuggle Deployment Script
# Deploys VectorSmuggle to Docker Compose or Kubernetes environments

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENVIRONMENT="${ENVIRONMENT:-development}"
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
VectorSmuggle Deployment Script

Usage: $0 [OPTIONS]

Options:
    -h, --help                    Show this help message
    -e, --environment ENV         Set environment (development|production) (default: development)
    -p, --platform PLATFORM      Set platform (docker-compose|kubernetes) (default: docker-compose)
    -n, --namespace NAMESPACE     Set Kubernetes namespace (default: vectorsmuggle)
    --profile PROFILE            Docker Compose profile to use
    --build                      Build images before deployment
    --update                     Update existing deployment
    --rollback                   Rollback to previous version
    --scale REPLICAS             Scale to specified number of replicas
    --dry-run                    Show what would be deployed without executing

Environments:
    development                  Development environment with debug features
    production                   Production environment with optimizations

Platforms:
    docker-compose              Deploy using Docker Compose
    kubernetes                  Deploy to Kubernetes cluster

Examples:
    $0                                           # Deploy to development with Docker Compose
    $0 --environment production --platform kubernetes  # Deploy to production Kubernetes
    $0 --profile dev --build                    # Build and deploy development profile
    $0 --platform kubernetes --dry-run          # Show Kubernetes manifests without applying

EOF
}

# Parse command line arguments
BUILD=false
UPDATE=false
ROLLBACK=false
DRY_RUN=false
SCALE=""
PROFILE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -p|--platform)
            PLATFORM="$2"
            shift 2
            ;;
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        --profile)
            PROFILE="$2"
            shift 2
            ;;
        --build)
            BUILD=true
            shift
            ;;
        --update)
            UPDATE=true
            shift
            ;;
        --rollback)
            ROLLBACK=true
            shift
            ;;
        --scale)
            SCALE="$2"
            shift 2
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

# Validate environment
validate_environment() {
    case $ENVIRONMENT in
        development|production)
            ;;
        *)
            log_error "Invalid environment: $ENVIRONMENT"
            log_error "Valid environments: development, production"
            exit 1
            ;;
    esac
    
    case $PLATFORM in
        docker-compose|kubernetes)
            ;;
        *)
            log_error "Invalid platform: $PLATFORM"
            log_error "Valid platforms: docker-compose, kubernetes"
            exit 1
            ;;
    esac
}

# Check dependencies
check_dependencies() {
    log_info "Checking dependencies for $PLATFORM deployment..."
    
    if [[ "$PLATFORM" == "docker-compose" ]]; then
        if ! command -v docker &> /dev/null; then
            log_error "Docker is not installed or not in PATH"
            exit 1
        fi
        
        if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
            log_error "Docker Compose is not installed or not in PATH"
            exit 1
        fi
        
        if ! docker info &> /dev/null; then
            log_error "Docker daemon is not running"
            exit 1
        fi
    elif [[ "$PLATFORM" == "kubernetes" ]]; then
        if ! command -v kubectl &> /dev/null; then
            log_error "kubectl is not installed or not in PATH"
            exit 1
        fi
        
        if ! kubectl cluster-info &> /dev/null; then
            log_error "Cannot connect to Kubernetes cluster"
            exit 1
        fi
        
        if ! command -v helm &> /dev/null; then
            log_warning "Helm is not installed. Some features may not be available."
        fi
    fi
    
    log_success "Dependencies check passed"
}

# Build images if requested
build_images() {
    if [[ "$BUILD" != true ]]; then
        return 0
    fi
    
    log_info "Building images..."
    
    BUILD_SCRIPT="${SCRIPT_DIR}/build.sh"
    if [[ -f "$BUILD_SCRIPT" ]]; then
        if [[ "$ENVIRONMENT" == "development" ]]; then
            "$BUILD_SCRIPT" --dev
        else
            "$BUILD_SCRIPT" --prod --scan
        fi
    else
        log_error "Build script not found: $BUILD_SCRIPT"
        exit 1
    fi
}

# Deploy with Docker Compose
deploy_docker_compose() {
    log_info "Deploying with Docker Compose..."
    
    cd "$PROJECT_ROOT"
    
    # Determine compose files
    COMPOSE_FILES=("-f" "docker-compose.yml")
    
    if [[ "$ENVIRONMENT" == "development" ]]; then
        COMPOSE_FILES+=("-f" "docker-compose.dev.yml")
        PROFILE="${PROFILE:-dev}"
    elif [[ "$ENVIRONMENT" == "production" ]]; then
        COMPOSE_FILES+=("-f" "docker-compose.prod.yml")
        PROFILE="${PROFILE:-full}"
    fi
    
    # Set environment variables
    export COMPOSE_PROJECT_NAME="vectorsmuggle-${ENVIRONMENT}"
    export VERSION="${VERSION:-latest}"
    export BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
    export VCS_REF="$(git rev-parse HEAD 2>/dev/null || echo 'unknown')"
    
    # Compose command
    if command -v docker-compose &> /dev/null; then
        COMPOSE_CMD="docker-compose"
    else
        COMPOSE_CMD="docker compose"
    fi
    
    # Profile arguments
    PROFILE_ARGS=()
    if [[ -n "$PROFILE" ]]; then
        PROFILE_ARGS=("--profile" "$PROFILE")
    fi
    
    if [[ "$DRY_RUN" == true ]]; then
        log_info "Dry run - showing configuration:"
        $COMPOSE_CMD "${COMPOSE_FILES[@]}" "${PROFILE_ARGS[@]}" config
        return 0
    fi
    
    if [[ "$ROLLBACK" == true ]]; then
        log_info "Rolling back deployment..."
        $COMPOSE_CMD "${COMPOSE_FILES[@]}" "${PROFILE_ARGS[@]}" down
        # In a real scenario, you'd restore from backup or previous version
        log_warning "Rollback completed. Manual intervention may be required."
        return 0
    fi
    
    if [[ "$UPDATE" == true ]]; then
        log_info "Updating existing deployment..."
        $COMPOSE_CMD "${COMPOSE_FILES[@]}" "${PROFILE_ARGS[@]}" pull
        $COMPOSE_CMD "${COMPOSE_FILES[@]}" "${PROFILE_ARGS[@]}" up -d --remove-orphans
    else
        log_info "Starting new deployment..."
        $COMPOSE_CMD "${COMPOSE_FILES[@]}" "${PROFILE_ARGS[@]}" up -d --remove-orphans
    fi
    
    # Scale if requested
    if [[ -n "$SCALE" ]]; then
        log_info "Scaling vectorsmuggle service to $SCALE replicas..."
        $COMPOSE_CMD "${COMPOSE_FILES[@]}" "${PROFILE_ARGS[@]}" up -d --scale vectorsmuggle="$SCALE"
    fi
    
    # Show status
    log_info "Deployment status:"
    $COMPOSE_CMD "${COMPOSE_FILES[@]}" "${PROFILE_ARGS[@]}" ps
    
    log_success "Docker Compose deployment completed"
}

# Deploy to Kubernetes
deploy_kubernetes() {
    log_info "Deploying to Kubernetes..."
    
    cd "$PROJECT_ROOT"
    
    # Create namespace if it doesn't exist
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_info "Creating namespace: $NAMESPACE"
        if [[ "$DRY_RUN" == true ]]; then
            echo "kubectl create namespace $NAMESPACE"
        else
            kubectl create namespace "$NAMESPACE"
        fi
    fi
    
    # Apply manifests
    MANIFESTS_DIR="k8s"
    
    if [[ "$DRY_RUN" == true ]]; then
        log_info "Dry run - showing manifests:"
        kubectl apply --dry-run=client -f "$MANIFESTS_DIR/" -n "$NAMESPACE"
        return 0
    fi
    
    if [[ "$ROLLBACK" == true ]]; then
        log_info "Rolling back Kubernetes deployment..."
        kubectl rollout undo deployment/vectorsmuggle -n "$NAMESPACE"
        kubectl rollout status deployment/vectorsmuggle -n "$NAMESPACE"
        log_success "Rollback completed"
        return 0
    fi
    
    # Apply ConfigMaps and Secrets first
    log_info "Applying ConfigMaps and Secrets..."
    kubectl apply -f "$MANIFESTS_DIR/configmap.yaml" -n "$NAMESPACE"
    kubectl apply -f "$MANIFESTS_DIR/secret.yaml" -n "$NAMESPACE"
    
    # Apply other manifests
    log_info "Applying deployments and services..."
    kubectl apply -f "$MANIFESTS_DIR/deployment.yaml" -n "$NAMESPACE"
    kubectl apply -f "$MANIFESTS_DIR/service.yaml" -n "$NAMESPACE"
    kubectl apply -f "$MANIFESTS_DIR/ingress.yaml" -n "$NAMESPACE"
    
    # Scale if requested
    if [[ -n "$SCALE" ]]; then
        log_info "Scaling deployment to $SCALE replicas..."
        kubectl scale deployment vectorsmuggle --replicas="$SCALE" -n "$NAMESPACE"
    fi
    
    # Wait for deployment to be ready
    log_info "Waiting for deployment to be ready..."
    kubectl rollout status deployment/vectorsmuggle -n "$NAMESPACE" --timeout=300s
    kubectl rollout status deployment/qdrant -n "$NAMESPACE" --timeout=300s
    
    # Show status
    log_info "Deployment status:"
    kubectl get pods -n "$NAMESPACE"
    kubectl get services -n "$NAMESPACE"
    
    log_success "Kubernetes deployment completed"
}

# Health check
run_health_check() {
    log_info "Running health checks..."
    
    HEALTH_SCRIPT="${SCRIPT_DIR}/health-check.sh"
    if [[ -f "$HEALTH_SCRIPT" ]]; then
        "$HEALTH_SCRIPT" --platform "$PLATFORM" --namespace "$NAMESPACE"
    else
        log_warning "Health check script not found: $HEALTH_SCRIPT"
    fi
}

# Main execution
main() {
    log_info "Starting VectorSmuggle deployment..."
    log_info "Environment: $ENVIRONMENT"
    log_info "Platform: $PLATFORM"
    
    if [[ "$PLATFORM" == "kubernetes" ]]; then
        log_info "Namespace: $NAMESPACE"
    fi
    
    validate_environment
    check_dependencies
    build_images
    
    if [[ "$PLATFORM" == "docker-compose" ]]; then
        deploy_docker_compose
    elif [[ "$PLATFORM" == "kubernetes" ]]; then
        deploy_kubernetes
    fi
    
    if [[ "$DRY_RUN" != true ]] && [[ "$ROLLBACK" != true ]]; then
        run_health_check
    fi
    
    log_success "Deployment process completed!"
    
    # Show access information
    if [[ "$DRY_RUN" != true ]]; then
        echo
        log_info "Access Information:"
        if [[ "$PLATFORM" == "docker-compose" ]]; then
            log_info "Application: http://localhost:8080"
            if [[ "$ENVIRONMENT" == "development" ]]; then
                log_info "Qdrant UI: http://localhost:6333/dashboard"
            fi
        elif [[ "$PLATFORM" == "kubernetes" ]]; then
            log_info "Use 'kubectl port-forward' to access services locally"
            log_info "kubectl port-forward -n $NAMESPACE svc/vectorsmuggle 8080:8080"
        fi
    fi
}

# Run main function
main "$@"