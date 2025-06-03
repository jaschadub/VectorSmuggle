#!/bin/bash

# VectorSmuggle Build Script
# Builds Docker images with proper tagging and security scanning

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
IMAGE_NAME="vectorsmuggle"
REGISTRY="${REGISTRY:-localhost:5000}"
VERSION="${VERSION:-$(git rev-parse --short HEAD 2>/dev/null || echo 'latest')}"
BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
VCS_REF="${VCS_REF:-$(git rev-parse HEAD 2>/dev/null || echo 'unknown')}"

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
VectorSmuggle Build Script

Usage: $0 [OPTIONS]

Options:
    -h, --help              Show this help message
    -v, --version VERSION   Set image version (default: git short hash or 'latest')
    -r, --registry REGISTRY Set container registry (default: localhost:5000)
    -t, --tag TAG          Additional tag for the image
    --no-cache             Build without using cache
    --scan                 Run security scan after build
    --push                 Push image to registry after build
    --dev                  Build development image
    --prod                 Build production image (default)

Examples:
    $0                                    # Build with default settings
    $0 --version v1.2.3 --push          # Build and push specific version
    $0 --dev --no-cache                  # Build dev image without cache
    $0 --scan --push                     # Build, scan, and push

EOF
}

# Parse command line arguments
ADDITIONAL_TAGS=()
NO_CACHE=""
SCAN=false
PUSH=false
TARGET="runtime"

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -v|--version)
            VERSION="$2"
            shift 2
            ;;
        -r|--registry)
            REGISTRY="$2"
            shift 2
            ;;
        -t|--tag)
            ADDITIONAL_TAGS+=("$2")
            shift 2
            ;;
        --no-cache)
            NO_CACHE="--no-cache"
            shift
            ;;
        --scan)
            SCAN=true
            shift
            ;;
        --push)
            PUSH=true
            shift
            ;;
        --dev)
            TARGET="builder"
            shift
            ;;
        --prod)
            TARGET="runtime"
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
check_dependencies() {
    log_info "Checking dependencies..."
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi
    
    if [[ "$SCAN" == true ]] && ! command -v trivy &> /dev/null; then
        log_warning "Trivy not found. Security scanning will be skipped."
        SCAN=false
    fi
    
    log_success "Dependencies check passed"
}

# Run code quality checks
run_quality_checks() {
    log_info "Running code quality checks..."
    
    cd "$PROJECT_ROOT"
    
    # Check if virtual environment exists
    if [[ ! -d ".venv" ]]; then
        log_warning "Virtual environment not found. Creating one..."
        python3 -m venv .venv
    fi
    
    # Activate virtual environment
    source .venv/bin/activate
    
    # Install development dependencies
    if [[ -f "requirements.txt" ]]; then
        pip install -q -r requirements.txt
    fi
    
    # Run ruff checks
    if command -v ruff &> /dev/null; then
        log_info "Running ruff linting..."
        if ! ruff check .; then
            log_error "Ruff linting failed"
            exit 1
        fi
        log_success "Ruff linting passed"
    fi
    
    # Run bandit security checks
    if command -v bandit &> /dev/null; then
        log_info "Running bandit security checks..."
        if ! bandit -r . -f json -o bandit-report.json; then
            log_warning "Bandit found security issues. Check bandit-report.json"
        else
            log_success "Bandit security checks passed"
        fi
    fi
    
    deactivate
}

# Build Docker image
build_image() {
    log_info "Building Docker image..."
    
    cd "$PROJECT_ROOT"
    
    # Construct image tags
    FULL_IMAGE_NAME="${REGISTRY}/${IMAGE_NAME}:${VERSION}"
    LATEST_IMAGE_NAME="${REGISTRY}/${IMAGE_NAME}:latest"
    
    # Build arguments
    BUILD_ARGS=(
        --build-arg "BUILD_DATE=${BUILD_DATE}"
        --build-arg "VERSION=${VERSION}"
        --build-arg "VCS_REF=${VCS_REF}"
        --target "${TARGET}"
        --tag "${FULL_IMAGE_NAME}"
        --tag "${LATEST_IMAGE_NAME}"
    )
    
    # Add additional tags
    for tag in "${ADDITIONAL_TAGS[@]}"; do
        BUILD_ARGS+=(--tag "${REGISTRY}/${IMAGE_NAME}:${tag}")
    done
    
    # Add no-cache if specified
    if [[ -n "$NO_CACHE" ]]; then
        BUILD_ARGS+=($NO_CACHE)
    fi
    
    # Build the image
    if docker build "${BUILD_ARGS[@]}" .; then
        log_success "Docker image built successfully"
        log_info "Image: ${FULL_IMAGE_NAME}"
        log_info "Size: $(docker images --format "table {{.Size}}" "${FULL_IMAGE_NAME}" | tail -n 1)"
    else
        log_error "Docker build failed"
        exit 1
    fi
}

# Run security scan
run_security_scan() {
    if [[ "$SCAN" != true ]]; then
        return 0
    fi
    
    log_info "Running security scan with Trivy..."
    
    FULL_IMAGE_NAME="${REGISTRY}/${IMAGE_NAME}:${VERSION}"
    
    # Run Trivy scan
    if trivy image --exit-code 1 --severity HIGH,CRITICAL "${FULL_IMAGE_NAME}"; then
        log_success "Security scan passed"
    else
        log_error "Security scan found vulnerabilities"
        log_warning "Consider updating base image or dependencies"
        # Don't exit on scan failure in development
        if [[ "$TARGET" == "runtime" ]]; then
            exit 1
        fi
    fi
}

# Push image to registry
push_image() {
    if [[ "$PUSH" != true ]]; then
        return 0
    fi
    
    log_info "Pushing image to registry..."
    
    FULL_IMAGE_NAME="${REGISTRY}/${IMAGE_NAME}:${VERSION}"
    LATEST_IMAGE_NAME="${REGISTRY}/${IMAGE_NAME}:latest"
    
    # Push versioned image
    if docker push "${FULL_IMAGE_NAME}"; then
        log_success "Pushed ${FULL_IMAGE_NAME}"
    else
        log_error "Failed to push ${FULL_IMAGE_NAME}"
        exit 1
    fi
    
    # Push latest image
    if docker push "${LATEST_IMAGE_NAME}"; then
        log_success "Pushed ${LATEST_IMAGE_NAME}"
    else
        log_error "Failed to push ${LATEST_IMAGE_NAME}"
        exit 1
    fi
    
    # Push additional tags
    for tag in "${ADDITIONAL_TAGS[@]}"; do
        TAG_IMAGE_NAME="${REGISTRY}/${IMAGE_NAME}:${tag}"
        if docker push "${TAG_IMAGE_NAME}"; then
            log_success "Pushed ${TAG_IMAGE_NAME}"
        else
            log_error "Failed to push ${TAG_IMAGE_NAME}"
            exit 1
        fi
    done
}

# Cleanup function
cleanup() {
    log_info "Cleaning up..."
    # Remove dangling images
    docker image prune -f &> /dev/null || true
}

# Main execution
main() {
    log_info "Starting VectorSmuggle build process..."
    log_info "Version: ${VERSION}"
    log_info "Registry: ${REGISTRY}"
    log_info "Target: ${TARGET}"
    
    check_dependencies
    run_quality_checks
    build_image
    run_security_scan
    push_image
    cleanup
    
    log_success "Build process completed successfully!"
    
    # Display final image information
    FULL_IMAGE_NAME="${REGISTRY}/${IMAGE_NAME}:${VERSION}"
    echo
    log_info "Built image: ${FULL_IMAGE_NAME}"
    log_info "Image ID: $(docker images --format "{{.ID}}" "${FULL_IMAGE_NAME}" | head -n 1)"
    log_info "Created: $(docker images --format "{{.CreatedAt}}" "${FULL_IMAGE_NAME}" | head -n 1)"
    
    if [[ "$PUSH" == true ]]; then
        log_info "Image pushed to registry and ready for deployment"
    else
        log_info "Use --push flag to push to registry"
    fi
}

# Trap cleanup on exit
trap cleanup EXIT

# Run main function
main "$@"