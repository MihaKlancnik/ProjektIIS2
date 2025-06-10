#!/bin/bash

# Crypto Predictor Deployment Script
# This script helps deploy the application locally or in production

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
ENVIRONMENT="development"
BUILD_IMAGE=false
PUSH_IMAGE=false
REGISTRY="ghcr.io"
IMAGE_NAME="crypto-predictor"

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show help
show_help() {
    cat << EOF
Crypto Predictor Deployment Script

Usage: $0 [OPTIONS]

OPTIONS:
    -e, --environment ENV    Deployment environment (development|staging|production) [default: development]
    -b, --build             Build Docker image
    -p, --push              Push Docker image to registry
    -r, --registry URL      Docker registry URL [default: ghcr.io]
    -i, --image NAME        Docker image name [default: crypto-predictor]
    -h, --help              Show this help message

Examples:
    $0 -e development -b                    # Build for development
    $0 -e production -b -p                  # Build and push for production
    $0 -e staging --build --push            # Build and push for staging

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -b|--build)
            BUILD_IMAGE=true
            shift
            ;;
        -p|--push)
            PUSH_IMAGE=true
            shift
            ;;
        -r|--registry)
            REGISTRY="$2"
            shift 2
            ;;
        -i|--image)
            IMAGE_NAME="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate environment
if [[ ! "$ENVIRONMENT" =~ ^(development|staging|production)$ ]]; then
    print_error "Invalid environment: $ENVIRONMENT"
    print_error "Must be one of: development, staging, production"
    exit 1
fi

print_status "Starting deployment for environment: $ENVIRONMENT"

# Set image tag based on environment
if [ "$ENVIRONMENT" = "development" ]; then
    IMAGE_TAG="dev"
elif [ "$ENVIRONMENT" = "staging" ]; then
    IMAGE_TAG="staging"
else
    IMAGE_TAG="latest"
fi

FULL_IMAGE_NAME="$REGISTRY/$IMAGE_NAME:$IMAGE_TAG"

# Build Docker image if requested
if [ "$BUILD_IMAGE" = true ]; then
    print_status "Building Docker image: $FULL_IMAGE_NAME"
    docker build -t "$FULL_IMAGE_NAME" .
    print_status "Docker image built successfully"
fi

# Push Docker image if requested
if [ "$PUSH_IMAGE" = true ]; then
    print_status "Pushing Docker image: $FULL_IMAGE_NAME"
    docker push "$FULL_IMAGE_NAME"
    print_status "Docker image pushed successfully"
fi

# Deploy based on environment
case $ENVIRONMENT in
    development)
        print_status "Starting development environment with docker-compose"
        docker-compose up -d crypto-predictor
        print_status "Application available at: http://localhost:5000"
        ;;
    staging)
        print_status "Deploying to staging environment"
        # Add staging deployment logic here
        print_warning "Staging deployment requires server configuration"
        ;;
    production)
        print_status "Deploying to production environment"
        docker-compose --profile production up -d
        print_status "Production environment started with nginx reverse proxy"
        print_status "Application available at: http://localhost"
        ;;
esac

print_status "Deployment completed successfully!"

# Show running containers
print_status "Running containers:"
docker ps --filter "name=crypto-predictor"
