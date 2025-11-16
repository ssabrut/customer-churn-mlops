# Software Architecture Diagram

## System Architecture Overview

This document provides a comprehensive view of the Customer Churn MLOps platform architecture, including all components, data flows, and interactions.

## High-Level Architecture

```mermaid
graph TB
    subgraph "User Interface Layer"
        UI[Gradio UI<br/>Port: 7860]
    end
    
    subgraph "API Layer"
        API[FastAPI Server<br/>Port: 8000<br/>/api/v1/predict]
    end
    
    subgraph "ML Services Layer"
        MLFLOW[MLflow Server<br/>Port: 5050<br/>Model Registry & Tracking]
        FEAST[Feast Feature Store<br/>Feature Repository]
    end
    
    subgraph "Orchestration Layer"
        AIRFLOW_WS[Airflow Webserver<br/>Port: 8080]
        AIRFLOW_SCHED[Airflow Scheduler]
        DAG1[Training DAG<br/>Daily Schedule]
        DAG2[Ground Truth DAG<br/>Daily Schedule]
    end
    
    subgraph "Storage Layer"
        APP_DB[(App PostgreSQL<br/>Port: 5435<br/>Prediction Logs)]
        MLFLOW_DB[(MLflow PostgreSQL<br/>Port: 5432<br/>Experiments & Metadata)]
        AIRFLOW_DB[(Airflow PostgreSQL<br/>Port: 5436<br/>DAG Metadata)]
        REDIS[(Redis<br/>Port: 6379<br/>Feast Online Store)]
        MINIO[(MinIO/S3<br/>Port: 9002<br/>Model Artifacts)]
    end
    
    subgraph "Monitoring Layer"
        GRAFANA[Grafana<br/>Port: 3000<br/>Dashboards]
    end
    
    UI -->|HTTP| API
    API -->|Load Model| MLFLOW
    API -->|Get Features| FEAST
    API -->|Log Predictions| APP_DB
    FEAST -->|Online Features| REDIS
    MLFLOW -->|Metadata| MLFLOW_DB
    MLFLOW -->|Artifacts| MINIO
    AIRFLOW_WS -->|Manage| DAG1
    AIRFLOW_WS -->|Manage| DAG2
    AIRFLOW_SCHED -->|Execute| DAG1
    AIRFLOW_SCHED -->|Execute| DAG2
    DAG1 -->|Train Models| MLFLOW
    DAG1 -->|Materialize| FEAST
    DAG2 -->|Fetch Ground Truth| APP_DB
    DAG2 -->|Calculate Metrics| MLFLOW
    GRAFANA -->|Query| APP_DB
    GRAFANA -->|Query| MLFLOW_DB
    
    style UI fill:#e1f5ff
    style API fill:#4fc3f7
    style MLFLOW fill:#81c784
    style FEAST fill:#81c784
    style AIRFLOW_WS fill:#ffb74d
    style AIRFLOW_SCHED fill:#ffb74d
    style GRAFANA fill:#ba68c8
```

## Detailed Component Architecture

```mermaid
graph LR
    subgraph "External Access"
        USER[Users/Developers]
        CLI[CLI Tools]
    end
    
    subgraph "Frontend Services"
        GRADIO[Gradio Server<br/>Container: gradio_server<br/>Port: 7860]
    end
    
    subgraph "Application Services"
        FASTAPI[FastAPI Application<br/>Container: fastapi_server<br/>Port: 8000]
        ROUTER1[Churn Router<br/>/predict/:id]
        ROUTER2[Health Router<br/>/health]
    end
    
    subgraph "ML Services"
        MLFLOW_SVC[MLflow Server<br/>Container: mlflow<br/>Port: 5050]
        MLFLOW_CLIENT[MLflow Client<br/>Model Loading]
        FEAST_SVC[Feast FeatureStore<br/>Feature Retrieval]
    end
    
    subgraph "Orchestration Services"
        AF_WS[Airflow Webserver<br/>Container: airflow_webserver<br/>Port: 8080]
        AF_SCHED[Airflow Scheduler<br/>Container: airflow_scheduler]
        AF_INIT[Airflow Init<br/>Container: airflow_init]
    end
    
    subgraph "Data Processing"
        PREPROCESS[Preprocess Script<br/>scripts/preprocess_data.py]
        MATERIALIZE[Materialize Script<br/>scripts/feast_materialize.py]
        TRAIN[Train Script<br/>scripts/train.py]
        GT_FETCH[Fetch Ground Truth<br/>scripts/fetch_ground_truth.py]
        PERF_CALC[Calculate Performance<br/>scripts/calculate_performance.py]
    end
    
    subgraph "Databases"
        APP_PG[(App PostgreSQL<br/>Container: app_postgres<br/>Port: 5435)]
        MLFLOW_PG[(MLflow PostgreSQL<br/>Container: mlflow_postgres<br/>Port: 5432)]
        AF_PG[(Airflow PostgreSQL<br/>Container: airflow_postgres<br/>Port: 5436)]
    end
    
    subgraph "Storage & Cache"
        REDIS_CACHE[(Redis<br/>Container: redis<br/>Port: 6379)]
        MINIO_STORAGE[(MinIO<br/>Container: s3<br/>Port: 9002/9001)]
    end
    
    subgraph "Monitoring"
        GRAFANA_SVC[Grafana<br/>Container: grafana<br/>Port: 3000]
    end
    
    USER --> GRADIO
    CLI --> FASTAPI
    GRADIO -->|HTTP| FASTAPI
    FASTAPI --> ROUTER1
    FASTAPI --> ROUTER2
    ROUTER1 --> MLFLOW_CLIENT
    ROUTER1 --> FEAST_SVC
    ROUTER1 --> APP_PG
    MLFLOW_CLIENT --> MLFLOW_SVC
    FEAST_SVC --> REDIS_CACHE
    MLFLOW_SVC --> MLFLOW_PG
    MLFLOW_SVC --> MINIO_STORAGE
    AF_WS --> AF_PG
    AF_SCHED --> AF_PG
    AF_SCHED --> PREPROCESS
    AF_SCHED --> MATERIALIZE
    AF_SCHED --> TRAIN
    AF_SCHED --> GT_FETCH
    AF_SCHED --> PERF_CALC
    PREPROCESS --> APP_PG
    MATERIALIZE --> REDIS_CACHE
    MATERIALIZE --> APP_PG
    TRAIN --> MLFLOW_SVC
    TRAIN --> FEAST_SVC
    GT_FETCH --> APP_PG
    PERF_CALC --> APP_PG
    PERF_CALC --> MLFLOW_SVC
    GRAFANA_SVC --> APP_PG
    GRAFANA_SVC --> MLFLOW_PG
    
    style GRADIO fill:#e1f5ff
    style FASTAPI fill:#4fc3f7
    style MLFLOW_SVC fill:#81c784
    style FEAST_SVC fill:#81c784
    style AF_WS fill:#ffb74d
    style AF_SCHED fill:#ffb74d
    style GRAFANA_SVC fill:#ba68c8
```

## Data Flow Architecture

### Prediction Flow

```mermaid
sequenceDiagram
    participant User
    participant Gradio
    participant FastAPI
    participant Feast
    participant Redis
    participant MLflow
    participant MinIO
    participant AppDB
    
    User->>Gradio: Enter Customer ID
    Gradio->>FastAPI: GET /api/v1/predict/{id}
    FastAPI->>Feast: get_online_features(customer_id)
    Feast->>Redis: Retrieve cached features
    Redis-->>Feast: Return features
    Feast-->>FastAPI: Feature vector
    FastAPI->>MLflow: Load model (if not cached)
    MLflow->>MinIO: Fetch model artifacts
    MinIO-->>MLflow: Model file
    MLflow-->>FastAPI: Model pipeline
    FastAPI->>FastAPI: Predict churn
    FastAPI->>AppDB: Log prediction
    FastAPI-->>Gradio: Prediction result
    Gradio-->>User: Display result
```

### Training Pipeline Flow

```mermaid
sequenceDiagram
    participant Airflow
    participant Preprocess
    participant Materialize
    participant Train
    participant AppDB
    participant Redis
    participant Feast
    participant MLflow
    participant MinIO
    participant MLflowDB
    
    Airflow->>Preprocess: Trigger preprocess_data task
    Preprocess->>AppDB: Read raw data
    AppDB-->>Preprocess: Raw customer data
    Preprocess->>AppDB: Write preprocessed data
    Preprocess-->>Airflow: Task complete
    
    Airflow->>Materialize: Trigger feast_materialize task
    Materialize->>AppDB: Read preprocessed data
    AppDB-->>Materialize: Entity dataframe
    Materialize->>Feast: Materialize features
    Feast->>Redis: Write online features
    Materialize-->>Airflow: Task complete
    
    Airflow->>Train: Trigger train_model task
    Train->>Feast: get_historical_features()
    Feast-->>Train: Training dataset
    Train->>Train: Train XGBoost model
    Train->>MLflow: Start run & log metrics
    MLflow->>MLflowDB: Store experiment metadata
    MLflow->>MinIO: Store model artifacts
    Train->>MLflow: Register model version
    Train-->>Airflow: Task complete
```

### Ground Truth & Monitoring Flow

```mermaid
sequenceDiagram
    participant Airflow
    participant FetchGT
    participant CalcPerf
    participant AppDB
    participant MLflow
    participant MLflowDB
    participant Grafana
    
    Airflow->>FetchGT: Trigger join_ground_truth task
    FetchGT->>AppDB: Query predictions with dates
    AppDB-->>FetchGT: Prediction logs
    FetchGT->>AppDB: Join with actual churn data
    FetchGT->>AppDB: Update ground_truth column
    FetchGT-->>Airflow: Task complete
    
    Airflow->>CalcPerf: Trigger calculate_performance task
    CalcPerf->>AppDB: Query predictions with ground truth
    AppDB-->>CalcPerf: Labeled predictions
    CalcPerf->>CalcPerf: Calculate metrics (Accuracy, F1, ROC-AUC)
    CalcPerf->>MLflow: Log metrics to MLflow
    MLflow->>MLflowDB: Store performance metrics
    CalcPerf-->>Airflow: Task complete
    
    Grafana->>AppDB: Query prediction logs
    Grafana->>MLflowDB: Query model metrics
    AppDB-->>Grafana: Prediction data
    MLflowDB-->>Grafana: Performance metrics
    Grafana->>Grafana: Render dashboards
```

## Network Architecture

```mermaid
graph TB
    subgraph "Public Network"
        EXT[External Users]
        EXT -->|:7860| GRADIO
        EXT -->|:8000| FASTAPI
        EXT -->|:8080| AIRFLOW_WS
        EXT -->|:5050| MLFLOW
        EXT -->|:3000| GRAFANA
        EXT -->|:9001| MINIO_CONSOLE
    end
    
    subgraph "Internal Network"
        GRADIO[Gradio Server]
        FASTAPI[FastAPI Server]
        AIRFLOW_WS[Airflow Webserver]
        AIRFLOW_SCHED[Airflow Scheduler]
        MLFLOW[MLflow Server]
        FEAST[Feast FeatureStore]
        APP_DB[(App PostgreSQL)]
        MLFLOW_DB[(MLflow PostgreSQL)]
        AIRFLOW_DB[(Airflow PostgreSQL)]
        REDIS[(Redis)]
        MINIO[(MinIO)]
        GRAFANA[Grafana]
    end
    
    GRADIO -->|Internal| FASTAPI
    FASTAPI -->|Internal| FEAST
    FASTAPI -->|Internal| MLFLOW
    FASTAPI -->|Internal| APP_DB
    FEAST -->|Internal| REDIS
    MLFLOW -->|Internal| MLFLOW_DB
    MLFLOW -->|Internal| MINIO
    AIRFLOW_WS -->|Internal| AIRFLOW_DB
    AIRFLOW_SCHED -->|Internal| AIRFLOW_DB
    AIRFLOW_SCHED -->|Internal| APP_DB
    AIRFLOW_SCHED -->|Internal| MLFLOW
    AIRFLOW_SCHED -->|Internal| FEAST
    GRAFANA -->|Internal| APP_DB
    GRAFANA -->|Internal| MLFLOW_DB
    
    style EXT fill:#ffcdd2
    style GRADIO fill:#e1f5ff
    style FASTAPI fill:#4fc3f7
    style MLFLOW fill:#81c784
    style AIRFLOW_WS fill:#ffb74d
    style GRAFANA fill:#ba68c8
```

## Container Architecture

```mermaid
graph TB
    subgraph "Docker Compose Stack"
        subgraph "Application Containers"
            C1[fastapi_server<br/>churn-mlops-image:latest]
            C2[gradio_server<br/>churn-mlops-image:latest]
            C3[app_db_init<br/>churn-mlops-image:latest]
        end
        
        subgraph "ML Containers"
            C4[mlflow<br/>Custom MLflow Image]
        end
        
        subgraph "Orchestration Containers"
            C5[airflow_webserver<br/>apache/airflow:2.8.0]
            C6[airflow_scheduler<br/>apache/airflow:2.8.0]
            C7[airflow_init<br/>apache/airflow:2.8.0]
        end
        
        subgraph "Database Containers"
            C8[app_postgres<br/>postgres:16]
            C9[mlflow_postgres<br/>postgres:16]
            C10[airflow_postgres<br/>postgres:16]
        end
        
        subgraph "Storage Containers"
            C11[s3<br/>minio/minio:latest]
            C12[mc_setup<br/>minio/mc]
            C13[redis<br/>redis:6.2-alpine]
        end
        
        subgraph "Monitoring Containers"
            C14[grafana<br/>grafana/grafana:latest]
        end
    end
    
    C1 --> C8
    C1 --> C4
    C1 --> C13
    C2 --> C1
    C3 --> C8
    C4 --> C9
    C4 --> C11
    C5 --> C10
    C6 --> C10
    C7 --> C10
    C12 --> C11
    C14 --> C8
    C14 --> C9
```

## Component Details

### FastAPI Application
- **Container**: `fastapi_server`
- **Port**: 8000
- **Endpoints**:
  - `GET /api/v1/predict/{customer_id}` - Churn prediction
  - `GET /api/v1/health` - Health check
- **Dependencies**: MLflow (model loading), Feast (feature retrieval), PostgreSQL (prediction logging)
- **Lifespan**: Loads model and Feast store on startup

### MLflow Server
- **Container**: `mlflow`
- **Port**: 5050
- **Backend Store**: PostgreSQL (experiments, runs, metadata)
- **Artifact Store**: MinIO S3 (model files, artifacts)
- **Functions**: Model registry, experiment tracking, model versioning

### Feast Feature Store
- **Repository**: `feature_repo/`
- **Online Store**: Redis (real-time feature serving)
- **Offline Store**: PostgreSQL (historical features)
- **Features**: Customer demographics, transaction history, interaction metrics

### Airflow Orchestration
- **Webserver**: Port 8080 (UI)
- **Scheduler**: Executes DAGs
- **DAGs**:
  1. `churn_model_training_pipeline` - Daily training
  2. `churn_ground_truth_pipeline` - Daily monitoring

### Databases
- **App PostgreSQL** (Port 5435): Prediction logs, customer data, ground truth
- **MLflow PostgreSQL** (Port 5432): Experiment metadata, run history
- **Airflow PostgreSQL** (Port 5436): DAG metadata, task history

### Storage
- **Redis** (Port 6379): Feast online feature cache
- **MinIO** (Port 9002/9001): S3-compatible object storage for MLflow artifacts

### Monitoring
- **Grafana** (Port 3000): Dashboards for model performance and system metrics
- **Data Sources**: App PostgreSQL, MLflow PostgreSQL

## Technology Stack Summary

| Layer            | Technology             | Purpose                        |
| ---------------- | ---------------------- | ------------------------------ |
| Frontend         | Gradio                 | Interactive UI for predictions |
| API              | FastAPI                | REST API for model serving     |
| ML Framework     | XGBoost, scikit-learn  | Model training and inference   |
| Feature Store    | Feast                  | Feature management and serving |
| Model Registry   | MLflow                 | Model versioning and tracking  |
| Orchestration    | Apache Airflow         | Workflow automation            |
| Databases        | PostgreSQL             | Data persistence               |
| Cache            | Redis                  | Feature caching                |
| Storage          | MinIO                  | Model artifact storage         |
| Monitoring       | Grafana                | Metrics and dashboards         |
| Containerization | Docker, Docker Compose | Deployment and orchestration   |

## Deployment Architecture

All services are containerized and orchestrated via Docker Compose:
- **Networks**: `internal`, `public`, `feast_net`
- **Volumes**: Persistent storage for databases and MinIO
- **Health Checks**: All services include health check configurations
- **Dependencies**: Services start in correct order based on dependencies

