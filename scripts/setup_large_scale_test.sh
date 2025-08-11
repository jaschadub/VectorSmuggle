#!/bin/bash

# VectorSmuggle Large-Scale Test Setup Script
# Validates environment, checks system requirements, and configures optimal settings

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
DEFAULT_ENRON_PATH="/media/jascha/BKUP01/enron-emails/maildir/"
DEFAULT_EMAIL_COUNT=100000
DEFAULT_BATCH_SIZE=1000
DEFAULT_OUTPUT_DIR="./large_scale_results"

# System requirements (in MB and GB)
MIN_MEMORY_GB=8
RECOMMENDED_MEMORY_GB=16
MIN_DISK_SPACE_GB=50
RECOMMENDED_DISK_SPACE_GB=100

print_header() {
    echo -e "${BLUE}================================================${NC}"
    echo -e "${BLUE}                        ,,,${NC}"
    echo -e "${BLUE}                     .'    \`/\\_/\\${NC}"
    echo -e "${BLUE}                   .'       <@I@>${NC}"
    echo -e "${BLUE}        <((((((((((  )____(  \\./${NC}"
    echo -e "${BLUE}                   \\( \\(   \\(\\(${NC}"
    echo -e "${BLUE}                    \`-\"\`-\"  \" \"${NC}"
    echo -e "${BLUE}  VectorSmuggle Large-Scale Test Setup${NC}"
    echo -e "${BLUE}================================================${NC}"
    echo
}

print_section() {
    echo -e "${YELLOW}$1${NC}"
    echo "----------------------------------------"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# Function to convert bytes to human readable format
bytes_to_human() {
    local bytes=$1
    if [ $bytes -gt 1073741824 ]; then
        echo "$(( bytes / 1073741824 ))GB"
    elif [ $bytes -gt 1048576 ]; then
        echo "$(( bytes / 1048576 ))MB"
    else
        echo "${bytes}B"
    fi
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check system requirements
check_system_requirements() {
    print_section "System Requirements Check"
    
    local requirements_met=true
    
    # Check Python version
    if command_exists python3; then
        local python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
        local major_version=$(echo $python_version | cut -d'.' -f1)
        local minor_version=$(echo $python_version | cut -d'.' -f2)
        
        if [ "$major_version" -eq 3 ] && [ "$minor_version" -ge 11 ]; then
            print_success "Python $python_version (>= 3.11 required)"
        else
            print_error "Python $python_version found, but 3.11+ required"
            requirements_met=false
        fi
    else
        print_error "Python 3 not found"
        requirements_met=false
    fi
    
    # Check memory
    if command_exists free; then
        local total_memory_kb=$(free | grep '^Mem:' | awk '{print $2}')
        local total_memory_gb=$(( total_memory_kb / 1024 / 1024 ))
        
        if [ $total_memory_gb -ge $RECOMMENDED_MEMORY_GB ]; then
            print_success "Memory: ${total_memory_gb}GB (recommended: ${RECOMMENDED_MEMORY_GB}GB+)"
        elif [ $total_memory_gb -ge $MIN_MEMORY_GB ]; then
            print_warning "Memory: ${total_memory_gb}GB (minimum: ${MIN_MEMORY_GB}GB, recommended: ${RECOMMENDED_MEMORY_GB}GB+)"
        else
            print_error "Memory: ${total_memory_gb}GB (minimum ${MIN_MEMORY_GB}GB required)"
            requirements_met=false
        fi
    else
        print_warning "Cannot check memory (free command not available)"
    fi
    
    # Check disk space
    local available_space_kb=$(df . | tail -1 | awk '{print $4}')
    local available_space_gb=$(( available_space_kb / 1024 / 1024 ))
    
    if [ $available_space_gb -ge $RECOMMENDED_DISK_SPACE_GB ]; then
        print_success "Disk space: ${available_space_gb}GB available (recommended: ${RECOMMENDED_DISK_SPACE_GB}GB+)"
    elif [ $available_space_gb -ge $MIN_DISK_SPACE_GB ]; then
        print_warning "Disk space: ${available_space_gb}GB available (minimum: ${MIN_DISK_SPACE_GB}GB, recommended: ${RECOMMENDED_DISK_SPACE_GB}GB+)"
    else
        print_error "Disk space: ${available_space_gb}GB available (minimum ${MIN_DISK_SPACE_GB}GB required)"
        requirements_met=false
    fi
    
    # Check CPU cores
    if command_exists nproc; then
        local cpu_cores=$(nproc)
        if [ $cpu_cores -ge 8 ]; then
            print_success "CPU cores: $cpu_cores (optimal: 8+)"
        elif [ $cpu_cores -ge 4 ]; then
            print_warning "CPU cores: $cpu_cores (minimum: 4, optimal: 8+)"
        else
            print_warning "CPU cores: $cpu_cores (recommended: 4+)"
        fi
    else
        print_warning "Cannot check CPU cores (nproc command not available)"
    fi
    
    echo
    return $([ "$requirements_met" = true ] && echo 0 || echo 1)
}

# Validate Enron email archive
validate_enron_archive() {
    print_section "Enron Email Archive Validation"
    
    local enron_path="${1:-$DEFAULT_ENRON_PATH}"
    
    if [ ! -d "$enron_path" ]; then
        print_error "Enron email archive not found at: $enron_path"
        print_info "Please ensure the Enron email archive is available at the specified path"
        print_info "Expected structure: $enron_path/[person]/[folder]/[numbered_files]"
        return 1
    fi
    
    print_success "Enron archive directory found: $enron_path"
    
    # Check for expected structure
    local person_dirs=$(find "$enron_path" -maxdepth 1 -type d | wc -l)
    if [ $person_dirs -lt 2 ]; then  # At least 1 person directory (plus the root)
        print_error "Invalid Enron archive structure - no person directories found"
        return 1
    fi
    
    print_success "Found $((person_dirs - 1)) person directories"
    
    # Count total email files
    local email_count=0
    if command_exists find; then
        # Look for numbered files (typical Enron email format)
        email_count=$(find "$enron_path" -type f -name '[0-9]*' 2>/dev/null | wc -l)
    fi
    
    if [ $email_count -gt 0 ]; then
        print_success "Found approximately $email_count email files"
        if [ $email_count -lt $DEFAULT_EMAIL_COUNT ]; then
            print_warning "Available emails ($email_count) less than default test size ($DEFAULT_EMAIL_COUNT)"
            print_info "Consider reducing --email-count parameter"
        fi
    else
        print_warning "Could not count email files - archive may use different naming convention"
    fi
    
    echo
    return 0
}

# Check Python dependencies
check_dependencies() {
    print_section "Python Dependencies Check"
    
    # Check if virtual environment is activated
    if [ -z "${VIRTUAL_ENV:-}" ]; then
        print_warning "Virtual environment not detected"
        print_info "Recommended: source .venv/bin/activate"
    else
        print_success "Virtual environment active: $VIRTUAL_ENV"
    fi
    
    # Check required packages
    local required_packages=("psutil" "numpy" "openai")
    local missing_packages=()
    
    for package in "${required_packages[@]}"; do
        if python3 -c "import $package" 2>/dev/null; then
            print_success "Package available: $package"
        else
            print_error "Missing package: $package"
            missing_packages+=("$package")
        fi
    done
    
    if [ ${#missing_packages[@]} -gt 0 ]; then
        print_info "Install missing packages with: pip install ${missing_packages[*]}"
        return 1
    fi
    
    echo
    return 0
}

# Check environment configuration
check_environment() {
    print_section "Environment Configuration Check"
    
    # Check API keys
    if [ -n "${OPENAI_API_KEY:-}" ]; then
        print_success "OpenAI API key configured"
    else
        print_warning "OpenAI API key not set"
        print_info "Set with: export OPENAI_API_KEY='your-key-here'"
        
        # Check for Ollama fallback
        if command_exists ollama; then
            print_info "Ollama detected - can be used as fallback"
            if ollama list | grep -q "nomic-embed-text"; then
                print_success "Ollama nomic-embed-text model available"
            else
                print_info "Install Ollama model with: ollama pull nomic-embed-text:latest"
            fi
        else
            print_warning "No embedding provider configured (OpenAI or Ollama)"
        fi
    fi
    
    # Check other environment variables
    if [ -n "${ENRON_EMAIL_PATH:-}" ]; then
        print_success "ENRON_EMAIL_PATH configured: $ENRON_EMAIL_PATH"
    else
        print_info "ENRON_EMAIL_PATH not set, will use default: $DEFAULT_ENRON_PATH"
    fi
    
    echo
}

# Generate optimal configuration
generate_configuration() {
    print_section "Optimal Configuration Recommendations"
    
    local memory_gb=8
    if command_exists free; then
        local total_memory_kb=$(free | grep '^Mem:' | awk '{print $2}')
        memory_gb=$(( total_memory_kb / 1024 / 1024 ))
    fi
    
    local cpu_cores=4
    if command_exists nproc; then
        cpu_cores=$(nproc)
    fi
    
    # Calculate optimal batch size based on available memory
    local optimal_batch_size=$DEFAULT_BATCH_SIZE
    if [ $memory_gb -ge 32 ]; then
        optimal_batch_size=2000
    elif [ $memory_gb -ge 16 ]; then
        optimal_batch_size=1000
    elif [ $memory_gb -ge 8 ]; then
        optimal_batch_size=500
    else
        optimal_batch_size=250
    fi
    
    # Calculate recommended email count
    local recommended_email_count=$DEFAULT_EMAIL_COUNT
    if [ $memory_gb -lt 16 ]; then
        recommended_email_count=50000
        print_warning "Reduced email count recommended due to limited memory"
    fi
    
    echo "Based on your system specifications:"
    echo "  Memory: ${memory_gb}GB"
    echo "  CPU cores: $cpu_cores"
    echo
    echo "Recommended configuration:"
    echo "  --email-count $recommended_email_count"
    echo "  --batch-size $optimal_batch_size"
    echo "  --output-dir $DEFAULT_OUTPUT_DIR"
    echo
    
    # Generate sample command
    print_info "Sample command:"
    echo "python generate_large_scale_report.py \\"
    echo "    --enron-path '$DEFAULT_ENRON_PATH' \\"
    echo "    --email-count $recommended_email_count \\"
    echo "    --batch-size $optimal_batch_size \\"
    echo "    --output-dir '$DEFAULT_OUTPUT_DIR' \\"
    echo "    --seed 42"
    echo
}

# Create output directory
setup_output_directory() {
    print_section "Output Directory Setup"
    
    local output_dir="${1:-$DEFAULT_OUTPUT_DIR}"
    
    if [ ! -d "$output_dir" ]; then
        if mkdir -p "$output_dir"; then
            print_success "Created output directory: $output_dir"
        else
            print_error "Failed to create output directory: $output_dir"
            return 1
        fi
    else
        print_success "Output directory exists: $output_dir"
    fi
    
    # Create subdirectories
    local subdirs=("reports" "logs" "temp")
    for subdir in "${subdirs[@]}"; do
        local full_path="$output_dir/$subdir"
        if [ ! -d "$full_path" ]; then
            if mkdir -p "$full_path"; then
                print_success "Created subdirectory: $full_path"
            else
                print_warning "Failed to create subdirectory: $full_path"
            fi
        fi
    done
    
    # Set appropriate permissions
    chmod 755 "$output_dir"
    
    echo
    return 0
}

# Performance monitoring setup
setup_monitoring() {
    print_section "Performance Monitoring Setup"
    
    # Check for monitoring tools
    local monitoring_tools=("htop" "iotop" "nethogs")
    local available_tools=()
    
    for tool in "${monitoring_tools[@]}"; do
        if command_exists "$tool"; then
            available_tools+=("$tool")
            print_success "Monitoring tool available: $tool"
        fi
    done
    
    if [ ${#available_tools[@]} -eq 0 ]; then
        print_warning "No advanced monitoring tools found"
        print_info "Consider installing: sudo apt-get install htop iotop nethogs"
    else
        print_info "Monitor system during test with: ${available_tools[0]}"
    fi
    
    # Check system monitoring capabilities
    if [ -f "/proc/meminfo" ] && [ -f "/proc/stat" ]; then
        print_success "System monitoring capabilities available"
    else
        print_warning "Limited system monitoring capabilities"
    fi
    
    echo
}

# Main setup function
main() {
    local enron_path="$DEFAULT_ENRON_PATH"
    local output_dir="$DEFAULT_OUTPUT_DIR"
    local skip_validation=false
    local quiet=false
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --enron-path)
                enron_path="$2"
                shift 2
                ;;
            --output-dir)
                output_dir="$2"
                shift 2
                ;;
            --skip-validation)
                skip_validation=true
                shift
                ;;
            --quiet)
                quiet=true
                shift
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo
                echo "Options:"
                echo "  --enron-path PATH     Path to Enron email archive (default: $DEFAULT_ENRON_PATH)"
                echo "  --output-dir PATH     Output directory for results (default: $DEFAULT_OUTPUT_DIR)"
                echo "  --skip-validation     Skip Enron archive validation"
                echo "  --quiet               Minimal output"
                echo "  --help                Show this help message"
                echo
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
    
    if [ "$quiet" = false ]; then
        print_header
    fi
    
    local setup_success=true
    
    # Run all checks
    if ! check_system_requirements; then
        setup_success=false
    fi
    
    if ! check_dependencies; then
        setup_success=false
    fi
    
    check_environment
    
    if [ "$skip_validation" = false ]; then
        if ! validate_enron_archive "$enron_path"; then
            setup_success=false
        fi
    fi
    
    if ! setup_output_directory "$output_dir"; then
        setup_success=false
    fi
    
    setup_monitoring
    
    if [ "$quiet" = false ]; then
        generate_configuration
    fi
    
    # Final status
    echo
    if [ "$setup_success" = true ]; then
        print_success "Setup completed successfully!"
        print_info "You can now run the large-scale test with the recommended configuration above"
    else
        print_error "Setup completed with issues"
        print_info "Please address the errors above before running the large-scale test"
        exit 1
    fi
}

# Run main function with all arguments
main "$@"