from datetime import datetime
import enum
from sqlalchemy import Enum
from extensions import db

class SubscriptionTier(enum.Enum):
    FREE = "free"
    BASIC = "basic"
    PRO = "pro"
    ENTERPRISE = "enterprise"

class OptimizationStatus(enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class HardwareTarget(enum.Enum):
    CORTEX_M4 = "cortex-m4"
    CORTEX_M7 = "cortex-m7"
    RISC_V = "risc-v"
    ARM_CORTEX_A = "arm-cortex-a"
    ESP32 = "esp32"
    STM32 = "stm32"

class ModelFramework(enum.Enum):
    TENSORFLOW = "tensorflow"
    PYTORCH = "pytorch"
    ONNX = "onnx"
    TFLITE = "tflite"

class User(db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)
    
    # Relationships
    projects = db.relationship('Project', backref='user', lazy=True, cascade='all, delete-orphan')
    subscription = db.relationship('Subscription', backref='user', uselist=False, cascade='all, delete-orphan')
    
    def to_dict(self):
        return {
            'id': self.id,
            'email': self.email,
            'username': self.username,
            'created_at': self.created_at.isoformat(),
            'is_active': self.is_active
        }

class Subscription(db.Model):
    __tablename__ = 'subscriptions'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    tier = db.Column(Enum(SubscriptionTier), nullable=False, default=SubscriptionTier.FREE)
    optimizations_used = db.Column(db.Integer, default=0)
    optimizations_limit = db.Column(db.Integer, default=5)  # Per month
    max_model_size_mb = db.Column(db.Integer, default=10)  # MB
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    expires_at = db.Column(db.DateTime)
    stripe_subscription_id = db.Column(db.String(255))
    
    def to_dict(self):
        return {
            'id': self.id,
            'tier': self.tier.value,
            'optimizations_used': self.optimizations_used,
            'optimizations_limit': self.optimizations_limit,
            'max_model_size_mb': self.max_model_size_mb,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None
        }

class Project(db.Model):
    __tablename__ = 'projects'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    name = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    optimization_jobs = db.relationship('OptimizationJob', backref='project', lazy=True, cascade='all, delete-orphan')
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'optimization_jobs_count': len(self.optimization_jobs)
        }

class OptimizationJob(db.Model):
    __tablename__ = 'optimization_jobs'
    
    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey('projects.id'), nullable=False)
    job_id = db.Column(db.String(36), unique=True, nullable=False)  # UUID
    
    # Model details
    original_model_path = db.Column(db.String(500), nullable=False)
    optimized_model_path = db.Column(db.String(500))
    model_framework = db.Column(Enum(ModelFramework), nullable=False)
    
    # Hardware target
    hardware_target = db.Column(Enum(HardwareTarget), nullable=False)
    target_size_kb = db.Column(db.Integer, default=100)  # Target size in KB
    target_accuracy_threshold = db.Column(db.Float, default=0.95)  # Minimum acceptable accuracy
    
    # Optimization settings
    enable_quantization = db.Column(db.Boolean, default=True)
    enable_pruning = db.Column(db.Boolean, default=True)
    enable_nas = db.Column(db.Boolean, default=False)  # More expensive
    quantization_type = db.Column(db.String(50), default='int8')  # int8, fp16, mixed
    pruning_sparsity = db.Column(db.Float, default=0.5)  # 0-1
    
    # Status and results
    status = db.Column(Enum(OptimizationStatus), default=OptimizationStatus.PENDING)
    progress_percentage = db.Column(db.Integer, default=0)
    error_message = db.Column(db.Text)
    
    # Timing
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    started_at = db.Column(db.DateTime)
    completed_at = db.Column(db.DateTime)
    
    # Results
    original_size_bytes = db.Column(db.Integer)
    optimized_size_bytes = db.Column(db.Integer)
    accuracy_before = db.Column(db.Float)
    accuracy_after = db.Column(db.Float)
    compression_ratio = db.Column(db.Float)
    
    # Relationships
    benchmark_results = db.relationship('BenchmarkResult', backref='optimization_job', lazy=True, cascade='all, delete-orphan')
    
    def to_dict(self):
        return {
            'id': self.id,
            'job_id': self.job_id,
            'model_framework': self.model_framework.value,
            'hardware_target': self.hardware_target.value,
            'target_size_kb': self.target_size_kb,
            'target_accuracy_threshold': self.target_accuracy_threshold,
            'status': self.status.value,
            'progress_percentage': self.progress_percentage,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'original_size_bytes': self.original_size_bytes,
            'optimized_size_bytes': self.optimized_size_bytes,
            'accuracy_before': self.accuracy_before,
            'accuracy_after': self.accuracy_after,
            'compression_ratio': self.compression_ratio,
            'error_message': self.error_message
        }

class BenchmarkResult(db.Model):
    __tablename__ = 'benchmark_results'
    
    id = db.Column(db.Integer, primary_key=True)
    optimization_job_id = db.Column(db.Integer, db.ForeignKey('optimization_jobs.id'), nullable=False)
    
    # Memory usage
    flash_usage_bytes = db.Column(db.Integer)
    ram_usage_bytes = db.Column(db.Integer)
    
    # Performance metrics
    inference_time_ms = db.Column(db.Float)
    cycles_per_inference = db.Column(db.Integer)
    power_consumption_mw = db.Column(db.Float)
    
    # Energy efficiency
    energy_per_inference_uj = db.Column(db.Float)  # microjoules
    
    # Additional metrics
    throughput_inferences_per_second = db.Column(db.Float)
    latency_percentile_95_ms = db.Column(db.Float)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'flash_usage_bytes': self.flash_usage_bytes,
            'ram_usage_bytes': self.ram_usage_bytes,
            'inference_time_ms': self.inference_time_ms,
            'cycles_per_inference': self.cycles_per_inference,
            'power_consumption_mw': self.power_consumption_mw,
            'energy_per_inference_uj': self.energy_per_inference_uj,
            'throughput_inferences_per_second': self.throughput_inferences_per_second,
            'latency_percentile_95_ms': self.latency_percentile_95_ms,
            'created_at': self.created_at.isoformat()
        }
