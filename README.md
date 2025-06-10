# Crypto Price Prediction MLOps Project

An end-to-end MLOps project for cryptocurrency price prediction using LSTM neural networks. The project includes automated data collection, validation, model training, and prediction deployment with automated Docker containerization and production deployment.

## 🚀 Features

- **Automated Data Pipeline**: Hourly data collection for Bitcoin, Ethereum, and Solana
- **Data Validation**: Great Expectations for data quality assurance
- **ML Models**: LSTM neural networks for multi-step price prediction
- **Web Interface**: Flask-based dashboard for real-time predictions
- **MLOps Integration**: MLflow for experiment tracking and model management
- **Containerization**: Docker and Docker Compose for deployment
- **CI/CD Pipeline**: Automated builds and deployments via GitHub Actions
- **Production Ready**: Nginx reverse proxy, health checks, and monitoring

## 🏗️ Architecture

```
Data Sources → DVC → Data Validation → Model Training → Prediction → Web App
     ↓              ↓                    ↓              ↓           ↓
  CoinGecko    Great Expectations    LSTM Models    CSV Storage   Flask
  Fear&Greed        ↓                    ↓              ↓           ↓
     API         MLflow            Model Registry   Predictions  Docker
```

## 🛠️ Quick Start

### Prerequisites

- Python 3.11+
- Poetry
- Docker & Docker Compose
- Git

### Local Development

1. **Clone the repository**

   ```bash
   git clone <your-repo-url>
   cd ProjektIIS2
   ```

2. **Install dependencies**

   ```bash
   poetry install
   poetry shell
   ```

3. **Configure environment**

   ```bash
   cp .env.template .env
   # Edit .env with your configuration
   ```

4. **Run the application**
   ```bash
   python src/app.py
   ```

### Docker Deployment

#### Development Environment

```bash
# Build and run with Docker Compose
docker-compose up -d crypto-predictor

# Or use the deployment script
./deploy.sh -e development -b
# Windows: .\deploy.ps1 -Environment development -Build
```

#### Production Environment

```bash
# Build and deploy to production
./deploy.sh -e production -b
# Windows: .\deploy.ps1 -Environment production -Build

# With nginx reverse proxy
docker-compose --profile production up -d
```

## 🔄 CI/CD Pipeline

The project includes automated Docker build and deployment pipeline:

### GitHub Actions Workflows

1. **Data Pipeline** (`.github/workflows/fetch_data.yml`)

   - Runs every hour
   - Fetches new data
   - Validates data quality
   - Trains models
   - Generates predictions

2. **Docker Build & Deploy** (`.github/workflows/docker-deploy.yml`)
   - Builds Docker images on push to main
   - Pushes to GitHub Container Registry
   - Deploys to staging and production environments
   - Includes security scanning with Trivy

### Required GitHub Secrets

Configure these secrets in your GitHub repository:

```
# DagsHub Integration
DAGSHUB_USER=your_dagshub_username
DAGSHUB_TOKEN=your_dagshub_token
DAGSHUB_ACCESS_KEY_ID=your_access_key
DAGSHUB_SECRET_ACCESS_KEY=your_secret_key

# Production Deployment
PRODUCTION_HOST=your_production_server_ip
PRODUCTION_USER=your_server_username
PRODUCTION_SSH_KEY=your_ssh_private_key

# Staging Deployment
STAGING_HOST=your_staging_server_ip
STAGING_USER=your_staging_username
STAGING_SSH_KEY=your_ssh_private_key

# GitHub Personal Access Token
PAT_TOKEN=your_github_pat_token
```

## 📊 Data Pipeline

### Data Sources

- **CoinGecko API**: Real-time cryptocurrency prices
- **Fear & Greed Index**: Market sentiment indicator

### Pipeline Steps

1. **Data Ingestion**: Automated hourly collection
2. **Data Validation**: Great Expectations checkpoints
3. **Feature Engineering**: Technical indicators and lag features
4. **Model Training**: LSTM models with/without sentiment data
5. **Prediction Generation**: 5-hour ahead price forecasts
6. **Model Monitoring**: Performance tracking via MLflow

## 🧠 Machine Learning Models

### Architecture

- **Input**: 24-hour sequences of price and technical features
- **Model**: LSTM neural networks with 50 units
- **Output**: 5-hour ahead price predictions
- **Variants**: With and without Fear & Greed Index

### Features

- Price lag features (1-24 hours)
- Rolling statistics (mean, std, min, max)
- Time-based features (hour, day of week)
- Optional: Fear & Greed Index sentiment

## 🌐 Web Application

### Features

- Real-time price predictions
- Interactive charts and visualizations
- Model accuracy comparisons
- Data validation reports
- Responsive design

### Endpoints

- `/`: Main dashboard
- `/reports/<report_name>`: Data validation reports
- `/validations/<validation_suite>`: Validation results

## 🐳 Docker Configuration

### Images

- **Base**: Python 3.11-slim
- **Production**: Multi-stage build with nginx
- **Registry**: GitHub Container Registry (ghcr.io)

### Services

- **crypto-predictor**: Main Flask application
- **nginx**: Reverse proxy (production profile)

### Volumes

- `./data:/app/data`: Persistent data storage
- `./models:/app/models`: Model artifacts
- `./predictions:/app/predictions`: Prediction outputs
- `./reports:/app/reports`: Validation reports

## 🔧 Configuration

### Environment Variables

```bash
# Flask
FLASK_ENV=production
FLASK_APP=wsgi.py

# MLflow
MLFLOW_TRACKING_USERNAME=your_username
MLFLOW_TRACKING_PASSWORD=your_token
DAGSHUB_USER=your_username
DAGSHUB_REPO=ProjektIIS2

# DVC
DAGSHUB_ACCESS_KEY_ID=your_key
DAGSHUB_SECRET_ACCESS_KEY=your_secret
```

### DVC Configuration

The project uses DVC for data versioning with DagsHub as remote storage.

## 📈 Monitoring & Logging

### Health Checks

- Application health endpoint
- Docker health checks
- Container restart policies

### Logging

- Structured logging with Python logging
- Container logs via Docker
- MLflow experiment tracking

### Security

- Trivy vulnerability scanning
- SARIF security reports
- GitHub Security tab integration

## 🚀 Production Deployment

### Server Requirements

- Docker and Docker Compose installed
- Minimum 2GB RAM, 1 CPU core
- 10GB disk space for data and models
- Network access to DagsHub and APIs

### Deployment Steps

1. Configure server with Docker
2. Set up GitHub secrets for deployment
3. Push to main branch to trigger deployment
4. Monitor deployment via GitHub Actions

### Scaling Considerations

- Use container orchestration (Kubernetes) for larger deployments
- Implement load balancing for multiple instances
- Consider using managed services for databases and storage

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Troubleshooting

### Common Issues

1. **Docker build fails**

   - Check Docker is running
   - Verify Dockerfile syntax
   - Ensure sufficient disk space

2. **Prediction errors**

   - Verify model files exist
   - Check data availability
   - Review application logs

3. **Deployment fails**
   - Verify GitHub secrets are set
   - Check server connectivity
   - Review SSH key permissions

### Support

For issues and questions:

- Check existing GitHub issues
- Create new issue with detailed description
- Include logs and error messages
