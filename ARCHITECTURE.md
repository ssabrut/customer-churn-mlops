# Customer Churn MLOps Architecture

This document captures the core architecture viewpoints for the Customer Churn prediction platform. It mirrors the structure of `ARCHITECTURE.md` while adding fully annotated diagrams and textual callouts for each view.

## High-Level Architecture

```mermaid
graph TB
    subgraph "User Entry Points"
        UI[Gradio UI<br/>Port 7860]
        API_DOCS[FastAPI Docs<br/>Port 8000]
    end

    subgraph "Application/Serving Layer"
        FASTAPI[FastAPI Service<br/>/api/v1/predict]
        ROUTERS[Prediction & Health Routers]
        FASTAPI -.-> ROUTERS
    end

    subgraph "ML Services Layer"
        FEAST[Feast Feature Store<br/>Feature Repo]
        MLFLOW[MLflow Tracking & Registry<br/>Port 5050]
    end

    subgraph "Orchestration Layer"
        AF_WS[Airflow Webserver<br/>Port 8080]
        AF_SCHED[Airflow Scheduler]
        TRAIN_DAG[Training DAG<br/>Daily]
        GT_DAG[Ground Truth DAG<br/>Daily]
    end

    subgraph "Storage Layer"
        APP_DB[(App PostgreSQL<br/>Port 5435)]
        MLFLOW_DB[(MLflow PostgreSQL<br/>Port 5432)]
        AIRFLOW_DB[(Airflow PostgreSQL<br/>Port 5436)]
        REDIS[(Redis Online Store<br/>Port 6379)]
        MINIO[(MinIO/S3<br/>Port 9002)]
    end

    subgraph "Monitoring Layer"
        GRAFANA[Grafana Dashboards<br/>Port 3000]
        PROM[Prometheus]
    end

    UI -->|HTTP| FASTAPI
    API_DOCS -->|HTTP| FASTAPI
    FASTAPI -->|REST Predict| MLFLOW
    FASTAPI -->|Online Features| FEAST
    FASTAPI -->|Log Predictions| APP_DB
    FEAST -->|Serve Features| REDIS
    MLFLOW -->|Artifacts| MINIO
    MLFLOW -->|Metadata| MLFLOW_DB
    AF_WS --> AF_SCHED
    AF_SCHED --> TRAIN_DAG
    AF_SCHED --> GT_DAG
    TRAIN_DAG -->|Materialize| FEAST
    TRAIN_DAG -->|Register Models| MLFLOW
    GT_DAG -->|Label Predictions| APP_DB
    GRAFANA -->|Query| APP_DB
    GRAFANA -->|Query| MLFLOW_DB
    PROM -->|Scrape| Services
```

**Key notes**
- Serving tier keeps an in-memory model handle loaded from MLflow on startup to reduce latency.
- Feast’s feature view definitions live in `feature_repo/definitions.py` with Redis as online store.
- Airflow orchestrates both model lifecycle and monitoring DAGs, sourcing data from PostgreSQL.

## Detailed Component Architecture

```mermaid
graph LR
    subgraph "External Interfaces"
        USER[Business Users]
        DEVOPS[Ops/ML Engineers]
    end

    subgraph "Frontend"
        GRADIO[gradio_server<br/>Port 7860]
    end

    subgraph "Serving"
        FASTAPI_SVC[fastapi_server<br/>Port 8000]
        ROUTER_HEALTH[Health Router<br/>/api/v1/health]
    end

    subgraph "ML Services"
        MLFLOW_SVC[mlflow<br/>Model Registry]
        MLFLOW_CLIENT[mlflow client<br/>`core/services/mlflow`]
        FEAST_SVC[feast feature store]
    end

    subgraph "Batch Processing Scripts"
        PREPROCESS[scripts/preprocess_data.py]
        MATERIALIZE[scripts/feast_materialize.py]
        TRAIN[scripts/train.py]
        FETCH_GT[scripts/fetch_ground_truth.py]
        PERF[scripts/calculate_performance.py]
        DRIFT[scripts/detect_drift.py]
    end

    subgraph "Airflow"
        AF_WS_SVC[airflow_webserver]
        AF_SCHED_SVC[airflow_scheduler]
        AF_DAGS[churn_*_pipeline DAGs]
    end

    subgraph "Stateful Stores"
        APP_DB[(app_postgres)]
        MLFLOW_DB[(mlflow_postgres)]
        AIRFLOW_DB[(airflow_postgres)]
        REDIS_CACHE[(redis)]
        MINIO_OBJ[(minio s3)]
    end

    subgraph "Observability"
        GRAFANA_SVC[grafana]
        PROM_SVC[prometheus]
    end

    USER --> GRADIO
    GRADIO --> FASTAPI_SVC
    FASTAPI_SVC --> ROUTER_PRED
    FASTAPI_SVC --> ROUTER_HEALTH
    ROUTER_PRED --> MLFLOW_CLIENT
    ROUTER_PRED --> FEAST_SVC
    ROUTER_PRED --> APP_DB
    MLFLOW_CLIENT --> MLFLOW_SVC
    FEAST_SVC --> REDIS_CACHE
    PREPROCESS --> APP_DB
    MATERIALIZE --> FEAST_SVC
    MATERIALIZE --> REDIS_CACHE
    TRAIN --> FEAST_SVC
    TRAIN --> MLFLOW_SVC
    FETCH_GT --> APP_DB
    PERF --> APP_DB
    PERF --> MLFLOW_SVC
    DRIFT --> APP_DB
    DRIFT --> FEAST_SVC
    AF_WS_SVC --> AF_DAGS
    AF_SCHED_SVC --> AF_DAGS
    AF_SCHED_SVC --> PREPROCESS
    AF_SCHED_SVC --> MATERIALIZE
    AF_SCHED_SVC --> TRAIN
    AF_SCHED_SVC --> FETCH_GT
    AF_SCHED_SVC --> PERF
    AF_SCHED_SVC --> DRIFT
    AF_SCHED_SVC --> AIRFLOW_DB
    MLFLOW_SVC --> MLFLOW_DB
    MLFLOW_SVC --> MINIO_OBJ
    GRAFANA_SVC --> APP_DB
    GRAFANA_SVC --> MLFLOW_DB
    PROM_SVC --> FASTAPI_SVC
    PROM_SVC --> GRADIO
    PROM_SVC --> Airflow
```

**Highlights**
- Scripts are containerized and triggered as Airflow tasks, sharing the same Docker image as FastAPI for dependency consistency.
- Redis serves both Feast online features and inference caching.
- Prometheus metrics feed Grafana dashboards for service SLOs.

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

    User->>Gradio: Input customer_id & request prediction
    Gradio->>FastAPI: GET /api/v1/predict/{id}
    FastAPI->>Feast: get_online_features(customer_id)
    Feast->>Redis: Retrieve feature vector
    Redis-->>Feast: Cached features
    Feast-->>FastAPI: Feature payload
    FastAPI->>MLflow: Load latest production model
    MLflow->>MinIO: Stream model artifacts
    MinIO-->>MLflow: Model files
    MLflow-->>FastAPI: Deserialized pipeline
    FastAPI->>FastAPI: Run preprocessing & inference
    FastAPI->>AppDB: Log prediction + metadata
    FastAPI-->>Gradio: Return churn probability & explanations
    Gradio-->>User: Render results
```

### Training Pipeline Flow

```mermaid
sequenceDiagram
    participant Airflow
    participant Preprocess
    participant Materialize
    participant Train
    participant Feast
    participant Redis
    participant AppDB
    participant MLflow
    participant MinIO
    participant MLflowDB

    Airflow->>Preprocess: Trigger preprocess_data task
    Preprocess->>AppDB: Extract raw labeled data
    AppDB-->>Preprocess: Raw tables
    Preprocess->>AppDB: Persist cleaned/parquet data
    Preprocess-->>Airflow: Task success

    Airflow->>Materialize: Trigger feast_materialize task
    Materialize->>AppDB: Pull entity dataframe
    Materialize->>Feast: Build feature registry snapshot
    Feast->>Redis: Push updated online features
    Materialize-->>Airflow: Task success

    Airflow->>Train: Trigger train task
    Train->>Feast: get_historical_features()
    Feast-->>Train: Offline training set
    Train->>Train: Fit XGBoost pipeline
    Train->>MLflow: Log params & metrics
    MLflow->>MLflowDB: Persist experiment metadata
    MLflow->>MinIO: Store artifacts
    Train->>MLflow: Register new model version / stage transition
    Train-->>Airflow: Task success
```

### Ground Truth & Monitoring Flow

```mermaid
sequenceDiagram
    participant Airflow
    participant FetchGT
    participant CalcPerf
    participant Drift
    participant AppDB
    participant MLflow
    participant MLflowDB
    participant Grafana

    Airflow->>FetchGT: Trigger join_ground_truth
    FetchGT->>AppDB: Retrieve prediction logs
    AppDB-->>FetchGT: Predictions & metadata
    FetchGT->>AppDB: Join with actual churn labels
    FetchGT->>AppDB: Update ground_truth column
    FetchGT-->>Airflow: Task success

    Airflow->>CalcPerf: Trigger calculate_performance
    CalcPerf->>AppDB: Pull labeled predictions
    AppDB-->>CalcPerf: Input dataset
    CalcPerf->>CalcPerf: Compute metrics + confusion matrix
    CalcPerf->>MLflow: Log monitoring run
    MLflow->>MLflowDB: Persist monitoring metrics
    CalcPerf-->>Airflow: Task success

    Airflow->>Drift: Trigger drift_detection DAG
    Drift->>AppDB: Pull recent inference samples
    Drift->>MLflow: Fetch reference statistics
    Drift->>Drift: KS/PSI tests
    Drift->>AppDB: Annotate drift flags
    Drift-->>Airflow: Task success

    Grafana->>AppDB: Query prediction + drift tables
    Grafana->>MLflowDB: Query experiment metrics
```

## Network Architecture

```mermaid
graph TB
    subgraph "Public Network (exposed ports)"
        USER_EXT[External User/Bastion]
        USER_EXT -->|7860| GRADIO_PUB[Gradio]
        USER_EXT -->|8000| FASTAPI_PUB[FastAPI]
        USER_EXT -->|8080| AF_WS_PUB[Airflow Webserver]
        USER_EXT -->|5050| MLFLOW_PUB[MLflow UI]
        USER_EXT -->|3000| GRAFANA_PUB[Grafana]
        USER_EXT -->|9001| MINIO_CONSOLE[MinIO Console]
    end

    subgraph "Internal Network (docker internal, feast_net)"
        GRADIO_INT[Gradio Container]
        FASTAPI_INT[FastAPI Container]
        AF_WS_INT[Airflow Webserver]
        AF_SCHED_INT[Airflow Scheduler]
        MLFLOW_INT[MLflow Service]
        FEAST_INT[Feast]
        APP_DB_INT[(App PostgreSQL)]
        MLFLOW_DB_INT[(MLflow PostgreSQL)]
        AIRFLOW_DB_INT[(Airflow PostgreSQL)]
        REDIS_INT[(Redis)]
        MINIO_INT[(MinIO S3)]
        GRAFANA_INT[Grafana]
        PROM_INT[Prometheus]
    end

    GRADIO_PUB -->|bridge| GRADIO_INT
    FASTAPI_PUB --> FASTAPI_INT
    AF_WS_PUB --> AF_WS_INT
    MLFLOW_PUB --> MLFLOW_INT
    GRAFANA_PUB --> GRAFANA_INT
    MINIO_CONSOLE --> MINIO_INT

    FASTAPI_INT --> FEAST_INT
    FASTAPI_INT --> MLFLOW_INT
    FASTAPI_INT --> APP_DB_INT
    FEAST_INT --> REDIS_INT
    MLFLOW_INT --> MLFLOW_DB_INT
    MLFLOW_INT --> MINIO_INT
    AF_SCHED_INT --> AIRFLOW_DB_INT
    AF_SCHED_INT --> APP_DB_INT
    AF_SCHED_INT --> FEAST_INT
    AF_SCHED_INT --> MLFLOW_INT
    GRAFANA_INT --> APP_DB_INT
    GRAFANA_INT --> MLFLOW_DB_INT
    PROM_INT --> FASTAPI_INT
    PROM_INT --> GRADIO_INT
```

**Security considerations**
- Internal services communicate on the Docker `internal` network; no direct exposure.
- Access to MLflow UI and Airflow UI is gated through authenticated ingress/Bastion.
- MinIO console (9001) requires access key/secret key supplied via environment variables.

## Container Architecture

```mermaid
graph TB
    subgraph "docker-compose stack"
        subgraph "Application"
            FASTAPI_C[fastapi_server<br/>churn-mlops-image]
            GRADIO_C[gradio_server<br/>churn-mlops-image]
            ENTRYPOINT_C[app_db_init]
        end

        subgraph "ML & Feature"
            MLFLOW_C[mlflow<br/>custom image]
        end

        subgraph "Airflow"
            AF_WS_C[airflow_webserver]
            AF_SCHED_C[airflow_scheduler]
            AF_INIT_C[airflow_init]
        end

        subgraph "Data Stores"
            APP_PG_C[app_postgres<br/>postgres:16]
            MLFLOW_PG_C[mlflow_postgres<br/>postgres:16]
            AIRFLOW_PG_C[airflow_postgres<br/>postgres:16]
            REDIS_C[redis:6.2]
            MINIO_C[minio:minio]
        end

        subgraph "Monitoring"
            GRAFANA_C[grafana]
            PROM_C[prometheus]
        end
    end

    FASTAPI_C --> APP_PG_C
    FASTAPI_C --> REDIS_C
    FASTAPI_C --> MLFLOW_C
    GRADIO_C --> FASTAPI_C
    ENTRYPOINT_C --> APP_PG_C
    MLFLOW_C --> MLFLOW_PG_C
    MLFLOW_C --> MINIO_C
    AF_WS_C --> AIRFLOW_PG_C
    AF_SCHED_C --> AIRFLOW_PG_C
    AF_SCHED_C --> FASTAPI_C
    AF_SCHED_C --> FEAST_JOB_C
    FEAST_JOB_C --> APP_PG_C
    FEAST_JOB_C --> REDIS_C
    GRAFANA_C --> APP_PG_C
    GRAFANA_C --> MLFLOW_PG_C
    PROM_C --> FASTAPI_C
    PROM_C --> GRADIO_C
```

**Runtime notes**
- Shared Docker image for FastAPI/Gradio ensures consistent Python environment.
- PostgreSQL and MinIO containers mount named volumes for durability; backup procedures snapshot these volumes nightly.
- Airflow containers mount `dags/` and `logs/` volumes for hot-reloads and task visibility.


