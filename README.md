# TinyML Model Optimization Pipeline - Backend

A cloud service for automatically optimizing neural networks for microcontroller deployment with size constraints under 100kB while maintaining acceptable accuracy and latency.

## Features

- **Multi-framework Support**: TensorFlow, PyTorch, ONNX, TensorFlow Lite
- **Hardware Targeting**: Cortex-M4, Cortex-M7, RISC-V, ESP32, STM32, ARM Cortex-A
- **Optimization Techniques**:
  - Post-training quantization (INT8, FP16, mixed precision)
  - Structured and unstructured pruning
  - Neural Architecture Search (NAS) for hardware-specific optimization
- **Comprehensive Benchmarking**: Flash usage, RAM usage, inference time, power consumption
- **Tiered Subscriptions**: Free, Basic, Pro, Enterprise with different limits
- **RESTful API**: Complete REST API for integration

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Flask API     │    │   Celery        │
│   (Upload UI)   │───▶│   (REST API)    │───▶│   (Background   │
│                 │    │                 │    │    Workers)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   SQLite/       │    │   Redis         │
                       │   PostgreSQL    │    │   (Job Queue)   │
                       │   (Database)    │    │                 │
                       └─────────────────┘    └─────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.8+
- Redis server
- SQLite (default) or PostgreSQL (production)

### Installation

1. **Clone and setup**:
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Configure environment**:
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. **Initialize database**:
```bash
python -c "from app import app, db; app.app_context().push(); db.create_all()"
```

4. **Start Redis** (if not already running):
```bash
redis-server
```

5. **Start Celery worker**:
```bash
celery -A celery_worker.celery worker --loglevel=info
```

6. **Start Flask development server**:
```bash
python app.py
```

The API will be available at `http://localhost:5000`

## API Endpoints

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API information |
| `GET` | `/health` | Health check |
| `GET` | `/api/docs` | API documentation |

### Projects

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/projects` | Create new project |
| `GET` | `/api/projects` | List user projects |
| `GET` | `/api/projects/{id}` | Get project details |

### Optimization Jobs

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/projects/{id}/optimize` | Start optimization job |
| `GET` | `/api/jobs/{job_id}` | Get job status |
| `GET` | `/api/jobs/{job_id}/results` | Get optimization results |
| `GET` | `/api/jobs/{job_id}/download` | Download optimized model |

### Configuration

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/hardware-targets` | List supported hardware |
| `GET` | `/api/model-frameworks` | List supported frameworks |
| `GET` | `/api/subscription` | Get subscription info |
| `POST` | `/api/subscription/upgrade` | Upgrade subscription |

### Statistics

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/stats` | Platform statistics |

## Usage Examples

### 1. Create a Project

```bash
curl -X POST http://localhost:5000/api/projects \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Image Classification Model",
    "description": "Optimizing ResNet for Cortex-M4"
  }'
```

### 2. Start Optimization Job

```bash
curl -X POST http://localhost:5000/api/projects/1/optimize \
  -F "model_file=@model.h5" \
  -F "hardware_target=cortex-m4" \
  -F "model_framework=tensorflow" \
  -F "target_size_kb=100" \
  -F "target_accuracy_threshold=0.90" \
  -F "enable_quantization=true" \
  -F "enable_pruning=true" \
  -F "quantization_type=int8" \
  -F "pruning_sparsity=0.5"
```

### 3. Check Job Status

```bash
curl http://localhost:5000/api/jobs/{job_id}
```

### 4. Get Results

```bash
curl http://localhost:5000/api/jobs/{job_id}/results
```

## Configuration

### Environment Variables

Key environment variables (see `.env.example`):

- `DATABASE_URL`: Database connection string
- `CELERY_BROKER_URL`: Redis URL for Celery
- `SECRET_KEY`: Flask secret key
- `UPLOAD_FOLDER`: Directory for uploaded models
- `CORS_ORIGINS`: Allowed CORS origins

### Subscription Tiers

| Tier | Optimizations/Month | Max Model Size | NAS | Priority |
|------|--------------------:|---------------:|:---:|:--------:|
| Free | 5 | 10 MB | ❌ | ❌ |
| Basic | 50 | 100 MB | ❌ | ❌ |
| Pro | 200 | 500 MB | ✅ | ✅ |
| Enterprise | Unlimited | Unlimited | ✅ | ✅ |

### Hardware Targets

Supported microcontroller targets:

- **ARM Cortex-M4**: 84MHz, 256KB RAM, 1MB Flash
- **ARM Cortex-M7**: 216MHz, 512KB RAM, 2MB Flash
- **RISC-V**: 100MHz, 128KB RAM, 512KB Flash
- **ESP32**: 240MHz, 520KB RAM, 4MB Flash
- **STM32**: 168MHz, 384KB RAM, 1.5MB Flash
- **ARM Cortex-A**: 1GHz, 1MB RAM, 8MB Flash

## Development

### Database Migrations

```bash
# Create migration
flask db migrate -m "Add new table"

# Apply migration
flask db upgrade
```

### Testing

```bash
python -m pytest tests/
```

### Code Structure

```
backend/
├── app.py                 # Flask application factory
├── models.py              # Database models
├── routes.py              # API routes
├── config.py              # Configuration classes
├── optimization_pipeline.py # Core optimization logic
├── celery_worker.py       # Celery worker setup
├── requirements.txt       # Python dependencies
├── .env.example          # Environment template
└── README.md             # This file
```

## Optimization Pipeline

The optimization process consists of several stages:

1. **Model Analysis**: Parse and analyze the uploaded model
2. **Quantization**: Reduce precision (FP32→INT8/FP16)
3. **Pruning**: Remove redundant connections
4. **Neural Architecture Search**: Find hardware-optimized architecture
5. **Benchmarking**: Generate performance reports

### Performance Metrics

The system generates comprehensive benchmarks:

- **Memory Usage**: Flash and RAM requirements
- **Performance**: Inference time, cycles per inference
- **Power**: Power consumption, energy per inference
- **Efficiency**: Throughput, latency percentiles

## Production Deployment

### Using Docker

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
```

### Using Docker Compose

```yaml
version: '3.8'
services:
  web:
    build: .
    ports:
      - "5000:5000"
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/tinyml
      - CELERY_BROKER_URL=redis://redis:6379
    depends_on:
      - db
      - redis

  worker:
    build: .
    command: celery -A celery_worker.celery worker --loglevel=info
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/tinyml
      - CELERY_BROKER_URL=redis://redis:6379
    depends_on:
      - db
      - redis

  db:
    image: postgres:13
    environment:
      POSTGRES_DB: tinyml
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass

  redis:
    image: redis:7-alpine
```

## Monitoring & Logging

- **Health Checks**: `/health` endpoint
- **Metrics**: `/api/stats` endpoint
- **Logging**: Structured logging with configurable levels
- **Error Tracking**: Integration ready for Sentry

## Security

- **File Upload Validation**: File type and size restrictions
- **Input Sanitization**: All inputs validated and sanitized
- **CORS Configuration**: Configurable CORS origins
- **Rate Limiting**: Ready for rate limiting implementation
- **Authentication**: JWT-ready authentication system

## License

MIT License - see LICENSE file for details.
