# Crypto Predictor Deployment Script for Windows
# This script helps deploy the application locally or in production

param(
    [Parameter()]
    [ValidateSet("development", "staging", "production")]
    [string]$Environment = "development",
    
    [Parameter()]
    [switch]$Build,
    
    [Parameter()]
    [switch]$Push,
    
    [Parameter()]
    [string]$Registry = "ghcr.io",
    
    [Parameter()]
    [string]$ImageName = "crypto-predictor",
    
    [Parameter()]
    [switch]$Help
)

# Function to show help
function Show-Help {
    Write-Host @"
Crypto Predictor Deployment Script for Windows

Usage: .\deploy.ps1 [OPTIONS]

OPTIONS:
    -Environment ENV    Deployment environment (development|staging|production) [default: development]
    -Build             Build Docker image
    -Push              Push Docker image to registry
    -Registry URL      Docker registry URL [default: ghcr.io]
    -ImageName NAME    Docker image name [default: crypto-predictor]
    -Help              Show this help message

Examples:
    .\deploy.ps1 -Environment development -Build
    .\deploy.ps1 -Environment production -Build -Push
    .\deploy.ps1 -Environment staging -Build -Push

"@
}

# Show help if requested
if ($Help) {
    Show-Help
    exit 0
}

# Function to print colored output
function Write-Status {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

Write-Status "Starting deployment for environment: $Environment"

# Set image tag based on environment
switch ($Environment) {
    "development" { $ImageTag = "dev" }
    "staging" { $ImageTag = "staging" }
    "production" { $ImageTag = "latest" }
}

$FullImageName = "$Registry/$ImageName`:$ImageTag"

# Build Docker image if requested
if ($Build) {
    Write-Status "Building Docker image: $FullImageName"
    docker build -t $FullImageName .
    if ($LASTEXITCODE -eq 0) {
        Write-Status "Docker image built successfully"
    } else {
        Write-Error "Failed to build Docker image"
        exit 1
    }
}

# Push Docker image if requested
if ($Push) {
    Write-Status "Pushing Docker image: $FullImageName"
    docker push $FullImageName
    if ($LASTEXITCODE -eq 0) {
        Write-Status "Docker image pushed successfully"
    } else {
        Write-Error "Failed to push Docker image"
        exit 1
    }
}

# Deploy based on environment
switch ($Environment) {
    "development" {
        Write-Status "Starting development environment with docker-compose"
        docker-compose up -d crypto-predictor
        Write-Status "Application available at: http://localhost:5000"
    }
    "staging" {
        Write-Status "Deploying to staging environment"
        Write-Warning "Staging deployment requires server configuration"
    }
    "production" {
        Write-Status "Deploying to production environment"
        docker-compose --profile production up -d
        Write-Status "Production environment started with nginx reverse proxy"
        Write-Status "Application available at: http://localhost"
    }
}

Write-Status "Deployment completed successfully!"

# Show running containers
Write-Status "Running containers:"
docker ps --filter "name=crypto-predictor"
