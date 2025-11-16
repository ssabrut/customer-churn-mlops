# Customer Churn Prediction - MLOps Platform

A production-ready MLOps platform for predicting customer churn using machine learning. This project demonstrates end-to-end ML lifecycle management with feature stores, model versioning, API serving, workflow orchestration, and monitoring.

## 🚀 Features

- **Feature Store**: Feast-based feature store with Redis online store for real-time feature serving
- **Model Management**: MLflow for experiment tracking, model registry, and versioning
- **API Serving**: FastAPI-based REST API for real-time predictions
- **Workflow Orchestration**: Apache Airflow DAGs for automated training and monitoring pipelines
- **Interactive UI**: Gradio interface for easy model interaction and testing
- **Monitoring**: Grafana dashboards for model performance and system metrics
- **Containerized**: Fully containerized with Docker and Docker Compose
- **Ground Truth Tracking**: Automated pipeline for collecting ground truth and calculating model performance metrics

## 🏗️ Architecture

```
┌─────────────────┐
│   Gradio UI     │  Port 7860
│   (Frontend)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  FastAPI Server │  Port 8000
│  (Prediction)   │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌─────────┐ ┌──────────┐
│  Feast  │ │  MLflow  │
│ Feature │ │  Model   │
│  Store  │ │ Registry │
└────┬────┘ └────┬─────┘
     │           │
┌────┴────┐ ┌────┴─────┐
│  Redis  │ │   MinIO  │
│ (Online)│ │ (S3-like)│
└─────────┘ └──────────┘
     │           │
     └─────┬─────┘
           ▼
    ┌─────────────┐
    │  PostgreSQL │
    │  (Multiple) │
    └─────────────┘
           │
           ▼
    ┌─────────────┐
    │   Airflow   │  Port 8080
    │ (Orchestr.) │
    └─────────────┘
```

## 🛠️ Tech Stack

- **Python**: 3.10
- **ML Framework**: XGBoost, scikit-learn
- **Feature Store**: Feast
- **Model Registry**: MLflow
- **API Framework**: FastAPI
- **Orchestration**: Apache Airflow
- **Databases**: PostgreSQL (multiple instances)
- **Caching**: Redis
- **Storage**: MinIO (S3-compatible)
- **UI**: Gradio
- **Monitoring**: Grafana
- **Containerization**: Docker, Docker Compose

## 📋 Prerequisites

- **Docker** (version 20.10+)
- **Docker Compose** (version 2.0+)
- **Make** (optional, for convenience commands)
- **Python 3.10** (for local development)
- **uv** (Python package manager)

## 🚦 Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd customer-churn-mlops
```

### 2. Environment Configuration

Create a `.env` file in the project root with the following variables:

```bash
# Environment Flag
IS_DOCKER=false  # Set to true when running in Docker

# MLflow Configuration
MLFLOW_TRACKING_URI=http://localhost:5050
MLFLOW_S3_ENDPOINT_URL=http://localhost:9002
AWS_ACCESS_KEY_ID=minioadmin
AWS_SECRET_ACCESS_KEY=minioadmin

# Application Database
APP_DB_USER=churn_user
APP_DB_PASSWORD=churn_password
APP_DB_NAME=churn_db
APP_DB_HOST=localhost
APP_DB_PORT=5435

# MLflow Database
MLFLOW_DB_USER=mlflow_user
MLFLOW_DB_PASSWORD=mlflow_password
MLFLOW_DB_NAME=mlflow_db

# Airflow Database
AIRFLOW_DB_USER=airflow
AIRFLOW_DB_PASSWORD=airflow
AIRFLOW_DB_NAME=airflow

# Feast Configuration
FEAST_REDIS_URL=redis://redis:6379

# FastAPI URL (for Gradio)
FASTAPI_URL=http://fastapi_server:8000
```

### 3. Deploy with Docker Compose

Using Make (recommended):

```bash
make deploy
```

Or manually:

```bash
# Apply Feast feature definitions
cd feature_repo && feast apply && cd ..

# Start all services
docker compose up -d --build
```

### 4. Access Services

Once deployed, access the services at:

- **FastAPI API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Gradio UI**: http://localhost:7860
- **MLflow UI**: http://localhost:5050
- **Airflow UI**: http://localhost:8080 (admin/admin)
- **Grafana**: http://localhost:3000 (admin/admin)
- **MinIO Console**: http://localhost:9001

### 5. Stop Services

```bash
make down
```

Or manually:

```bash
docker compose down -v
```

## 📁 Project Structure

```
customer-churn-mlops/
├── core/                    # Core application code
│   ├── routers/            # FastAPI route handlers
│   │   ├── churn.py       # Churn prediction endpoints
│   │   └── health.py      # Health check endpoints
│   ├── schemas/           # Pydantic models
│   ├── services/          # Service clients
│   │   ├── mlflow/        # MLflow client
│   │   └── postgres/      # PostgreSQL client
│   ├── utils/             # Utility functions
│   ├── config.py         # Configuration management
│   └── main.py           # FastAPI application
├── dags/                  # Airflow DAGs
│   ├── model_training_dag.py      # Training pipeline
│   └── ground_truth_dag.py        # Monitoring pipeline
├── scripts/               # Standalone scripts
│   ├── train.py          # Model training script
│   ├── preprocess_data.py # Data preprocessing
│   ├── feast_materialize.py # Feature materialization
│   ├── fetch_ground_truth.py # Ground truth collection
│   ├── calculate_performance.py # Performance metrics
│   └── populate_db.py    # Database initialization
├── feature_repo/         # Feast feature repository
│   ├── definitions.py    # Feature definitions
│   └── feature_store.yaml # Feast configuration
├── entrypoint/           # Application entry points
│   └── gradio_app.py    # Gradio UI application
├── notebooks/            # Jupyter notebooks for EDA
├── data/                 # Data files
│   ├── raw/             # Raw data
│   └── preprocessed/    # Preprocessed data
├── grafana/             # Grafana dashboards and provisioning
├── docker/              # Docker-related files
├── docker-compose.yml   # Docker Compose configuration
├── Dockerfile          # Main application Dockerfile
├── pyproject.toml      # Python project configuration
├── requirements.txt   # Python dependencies
└── Makefile           # Convenience commands
```

## 🔌 API Documentation

### Health Check

```bash
GET /api/v1/health
```

Returns the health status of the service and model.

### Predict Churn

```bash
GET /api/v1/predict/{customer_id}
```

Predicts churn probability for a given customer ID.

**Response:**
```json
{
  "prediction": 0,
  "probability": 0.23,
  "version": 1,
  "features": {
    "Age": 35,
    "Support Calls": 2,
    "Payment Delay": 0,
    "Total Spend": 1500.0,
    "Last Interaction": 5,
    "Male": 1,
    "Age_Group": "30-40",
    "Interaction_Frequency": "Medium"
  }
}
```

### Interactive API Documentation

Visit http://localhost:8000/docs for Swagger UI or http://localhost:8000/redoc for ReDoc.

## 🔄 Workflows

### Model Training Pipeline

The training pipeline (`churn_model_training_pipeline`) runs daily and includes:

1. **Preprocess Data**: Cleans and transforms raw data
2. **Feast Materialize**: Materializes features to the online store
3. **Train Model**: Trains XGBoost model and registers with MLflow

Trigger manually from Airflow UI or wait for scheduled execution.

### Ground Truth Pipeline

The monitoring pipeline (`churn_ground_truth_pipeline`) runs daily and includes:

1. **Join Ground Truth**: Collects actual churn labels
2. **Calculate Performance**: Computes model performance metrics

## 🧪 Usage Examples

### Using the API

```bash
# Predict churn for customer ID 1001
curl http://localhost:8000/api/v1/predict/1001
```

### Using Python Client

```python
import requests

response = requests.get("http://localhost:8000/api/v1/predict/1001")
prediction = response.json()
print(f"Churn Probability: {prediction['probability']:.2%}")
```

### Using Gradio UI

1. Navigate to http://localhost:7860
2. Enter a customer ID
3. Click "Predict" to get real-time predictions with features

## 🔧 Development

### Local Development Setup

1. **Install Dependencies:
   ```bash
   uv pip install -r requirements.txt
   ```

2. **Set Environment Variables**:
   - Copy `.env.example` to `.env` (if available)
   - Update `IS_DOCKER=false` for local development

3. **Run Services Locally**:
   - Start PostgreSQL, Redis, MinIO, and MLflow manually or via Docker Compose
   - Run FastAPI: `uvicorn core.main:app --reload`

### Running Scripts Locally

```bash
# Preprocess data
python scripts/preprocess_data.py

# Materialize features
python scripts/feast_materialize.py

# Train model
python scripts/train.py

# Fetch ground truth
python scripts/fetch_ground_truth.py --date 2025-01-01 --days_ago 30

# Calculate performance
python scripts/calculate_performance.py --date 2025-01-01 --days_ago 30
```

### Code Quality

The project uses:
- **Black** for code formatting
- **isort** for import sorting

Format code:
```bash
black .
isort .
```

## 📊 Monitoring

### Grafana Dashboards

Access Grafana at http://localhost:3000 to view:
- Model performance metrics
- Prediction logs
- System health metrics

### MLflow Tracking

Access MLflow at http://localhost:5050 to:
- View experiment runs
- Compare model versions
- Register production models
- Download model artifacts

### Airflow Monitoring

Access Airflow at http://localhost:8080 to:
- Monitor DAG execution
- View task logs
- Trigger manual runs
- Manage workflow schedules

## 🐛 Troubleshooting

### Common Issues

1. **Port Already in Use**
   - Check if ports 8000, 7860, 5050, 8080, 3000 are available
   - Modify ports in `docker-compose.yml` if needed

2. **Database Connection Errors**
   - Ensure all database services are healthy: `docker compose ps`
   - Check database credentials in `.env`

3. **Model Not Loading**
   - Verify MLflow service is running
   - Check if a model is registered in MLflow
   - Review FastAPI logs: `docker logs fastapi_server`

4. **Feast Store Errors**
   - Ensure Redis is running: `docker ps | grep redis`
   - Verify feature definitions: `cd feature_repo && feast apply`

### Viewing Logs

```bash
# View all logs
docker compose logs

# View specific service logs
docker compose logs fastapi_server
docker compose logs airflow_scheduler
docker compose logs mlflow
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the terms specified in the [LICENSE](LICENSE) file.

## 🙏 Acknowledgments

- Built with FastAPI, MLflow, Feast, and Apache Airflow
- Uses XGBoost for machine learning predictions

---

**Note**: This is a production-ready MLOps platform template. Ensure proper security configurations, secrets management, and production-grade infrastructure before deploying to production environments.
