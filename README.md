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

## TODO List

### 🚨 Critical for MVP (Must Complete Before Open Source)

#### Frontend Implementation
- [ ] **Complete NextJS frontend** - Currently only directory structure exists
  - [ ] Model upload component with drag-and-drop
  - [ ] Project dashboard with job status tracking
  - [ ] Results visualization with charts and metrics
  - [ ] Subscription management interface
  - [ ] Real-time progress updates via WebSocket

#### Core Optimization Engine
- [ ] **Replace simulation with real optimization algorithms**
  - [ ] Implement actual TensorFlow quantization using TF Lite
  - [ ] Implement PyTorch quantization using torch.quantization
  - [ ] Real model pruning implementation
  - [ ] Integrate with ONNX optimization tools
  - [ ] Real Neural Architecture Search (NAS) implementation

#### Authentication & User Management
- [ ] **User registration and login system**
  - [ ] JWT-based authentication
  - [ ] Password reset functionality
  - [ ] User profile management
  - [ ] Session management

#### Model Processing Pipeline
- [ ] **Real model analysis and conversion**
  - [ ] Model format validation and parsing
  - [ ] Framework-specific model loading
  - [ ] Model complexity analysis
  - [ ] Error handling for corrupted models

#### Hardware Benchmarking
- [ ] **Actual performance estimation**
  - [ ] Hardware-specific performance modeling
  - [ ] Real memory usage calculation
  - [ ] Power consumption estimation
  - [ ] Integration with hardware simulators

### 🔧 Important for MVP

#### Embedded SDK & Runtime
- [ ] **C/C++ runtime library**
  - [ ] Model inference engine for microcontrollers
  - [ ] Memory-optimized operations
  - [ ] Hardware-specific optimizations
  - [ ] Example projects for each platform

#### File Management
- [ ] **Proper file handling**
  - [ ] Secure file upload with validation
  - [ ] File storage organization
  - [ ] Model download functionality
  - [ ] Cleanup of temporary files

#### Testing & Quality
- [ ] **Comprehensive test suite**
  - [ ] Unit tests for all components
  - [ ] Integration tests for API
  - [ ] End-to-end testing
  - [ ] Performance testing

#### Documentation
- [ ] **Complete documentation**
  - [ ] API documentation with OpenAPI/Swagger
  - [ ] SDK documentation
  - [ ] Tutorial and examples
  - [ ] Deployment guide

### 🎯 Nice to Have for MVP

#### Advanced Features
- [ ] **Enhanced optimization options**
  - [ ] Custom optimization profiles
  - [ ] Multi-objective optimization
  - [ ] Accuracy vs size trade-off visualization
  - [ ] Batch optimization for multiple models

#### Monitoring & Analytics
- [ ] **Usage analytics and monitoring**
  - [ ] Job success/failure tracking
  - [ ] Performance metrics
  - [ ] User behavior analytics
  - [ ] Error reporting

#### Enterprise Features
- [ ] **Team collaboration**
  - [ ] Team workspaces
  - [ ] Model sharing
  - [ ] Role-based access control
  - [ ] Audit logging

### 📊 Current Implementation Status

| Component | Status | Completion |
|-----------|--------|------------|
| Backend API | ✅ Complete | 90% |
| Database Models | ✅ Complete | 95% |
| Frontend | ❌ Missing | 5% |
| Real Optimization | ❌ Simulated | 20% |
| Authentication | ❌ Missing | 0% |
| File Management | ⚠️ Basic | 30% |
| Embedded SDK | ❌ Missing | 0% |
| Testing | ⚠️ Minimal | 10% |
| Documentation | ⚠️ API Only | 40% |

### 🚀 Recommended MVP Development Order

1. **Phase 1: Core Functionality (4-6 weeks)**
   - Implement real optimization algorithms
   - Build basic frontend with upload/results
   - Add authentication system

2. **Phase 2: User Experience (3-4 weeks)**
   - Complete frontend dashboard
   - Add real-time progress tracking
   - Implement file management

3. **Phase 3: Hardware Integration (4-5 weeks)**
   - Develop embedded runtime
   - Create hardware templates
   - Add performance benchmarking

4. **Phase 4: Polish & Launch (2-3 weeks)**
   - Comprehensive testing
   - Documentation completion
   - Deployment optimization

### ⚡ Quick Wins for Demo

If you need something demonstrable quickly:
- [ ] Fix the missing imports in quantization.py (`hashlib`)
- [ ] Create a simple frontend upload form
- [ ] Add basic model file validation
- [ ] Implement simple TensorFlow Lite conversion
- [ ] Add progress indicators to the existing simulation

## License

MIT License - see LICENSE file for details.
