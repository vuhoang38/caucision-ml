# Caucision ML

This repository contains the **machine learning service** for Caucision, a thesis project from the Advanced Program in Computer Science at the University of Science, VNU-HCM (2023).

Given historical sales and promotion data, the service estimates the *incremental* effect of each promotion on each customer (via causal inference) and then recommends the optimal allocation of promotions under a budget constraint (via combinatorial optimization).

For the full thesis report, checkout this [google drive link](https://drive.google.com/file/d/1giZrbFcxN7sAxqUKHQ0J9UQRVzLqTA0B/view?usp=sharing).

## Problem

Marketing managers must decide *which promotion to send to which customer* while balancing customer preferences, business goals, and a finite budget. Caucision targets online retailers specifically and combines uplift modelling, causal inference, and optimization behind a friendly web interface.

The platform answers two questions:

1. **What is the impact of a specific promotion on a specific user?** — estimated with an **X-Learner** causal inference model on top of historical campaign data.
2. **What is the optimal promotion allocation that maximizes incremental purchases under a budget?** — solved with the **Offline Multiple Choice Knapsack Problem (Off-MCKP)** algorithm.

## Role of this service

Caucision is a three-tier system. This repo is the bottom tier:

```
┌────────────────────────┐
│  React Front-end       │  (web UI for marketers)
└──────────┬─────────────┘
           │ HTTPS
┌──────────▼─────────────┐
│  Rails API Gateway     │  (auth, business data, orchestration)
└──────────┬─────────────┘
           │ HTTP + RabbitMQ
┌──────────▼─────────────┐
│  Caucision ML  (this)  │  ← FastAPI + Celery + ScyllaDB
│  - Causal inference    │
│  - Optimization        │
└────────────────────────┘
```

The ML service exposes a small HTTP API and a Celery worker. It reads/writes large customer-level datasets in **ScyllaDB**, persists trained models and metadata in **PostgreSQL** (via the gateway), and consumes async jobs from **RabbitMQ** with **Redis** as the result backend.

## Features

- **Model training (`train_model` Celery task)** — trains an X-Learner on a project's uploaded dataset, with a user-chosen base regressor (Linear Regression, XGBoost, Random Forest, Tweedie, Decision Tree, SVR, MLP, or k-NN). Stores the fitted estimator, encoder, identified estimand and causal model as a pickled blob on the project, and writes per-user treatment-effect estimates to ScyllaDB.
- **Campaign-level inference (`POST /campaign_data`)** — given a new CSV of customers for an existing project, reuses the trained X-Learner to estimate effects for the new cohort and streams the result back as CSV.
- **Optimization (`POST /optimize`)** — given a campaign, per-promotion costs, and a budget, solves an Off-MCKP to return the best promotion (or no promotion) per customer.

## Tech stack

- **Web framework:** FastAPI + Uvicorn
- **Async jobs:** Celery + RabbitMQ (broker) + Redis (result backend)
- **Causal inference:** [DoWhy](https://github.com/py-why/dowhy), [EconML](https://github.com/py-why/EconML) (X-Learner meta-learner)
- **Base learners:** scikit-learn, XGBoost, LightGBM
- **Optimization:** custom Off-MCKP using `paretoset` + `sortedcontainers`
- **Storage:** PostgreSQL (via SQLAlchemy / `psycopg2`) and ScyllaDB (via `scylla-driver`)

## Configuration

Settings are loaded by `caucisionml/config.py` from `.env.test`, `.env.development`, `.env.production` (in that order of precedence). Required variables:

| Variable            | Purpose                                                  |
|---------------------|----------------------------------------------------------|
| `DATABASE_URL`      | PostgreSQL connection string for project/campaign data   |
| `SCYLLA_HOST`       | ScyllaDB host (defaults to local hostname)               |
| `SCYLLA_KEYSPACE`   | ScyllaDB keyspace (defaults to `caucision`)              |
| `REDIS_URL`         | Redis URL — Celery result backend                        |
| `CELERY_BROKER_URL` | RabbitMQ URL — Celery broker                             |
| `API_GATEWAY_URL`   | Base URL of the Rails gateway (used to post training results back) |

## Running locally

Install dependencies (Python 3.10):

```bash
pip install -r requirements.txt
```

Run the API:

```bash
uvicorn caucisionml.main:app --host 0.0.0.0 --port 8000
```

Run the Celery worker (in a separate shell — needs RabbitMQ + Redis reachable):

```bash
celery -A caucisionml.main.celery worker --loglevel=info
```

## Running with Docker

```bash
docker build -t caucision-ml -f Dockerfile .
docker build -t caucision-ml-worker -f Dockerfile.celery .
docker run --env-file .env.production -p 8000:8000 caucision-ml
docker run --env-file .env.production caucision-ml-worker
```

ScyllaDB, PostgreSQL, RabbitMQ, and Redis are expected to be provisioned separately.

