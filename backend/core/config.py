import os
from datetime import timedelta

class Config:
    """Base configuration class"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///tinyml_optimizer.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # File upload settings
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER') or 'uploads'
    MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB max file size
    
    # Celery configuration
    CELERY_BROKER_URL = os.environ.get('CELERY_BROKER_URL') or 'redis://localhost:6379'
    CELERY_RESULT_BACKEND = os.environ.get('CELERY_RESULT_BACKEND') or 'redis://localhost:6379'
    
    # Security settings
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY') or 'jwt-secret-change-in-production'
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=1)
    JWT_REFRESH_TOKEN_EXPIRES = timedelta(days=30)
    
    # CORS settings
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', '*').split(',')
    
    # Subscription tiers configuration
    SUBSCRIPTION_LIMITS = {
        'free': {
            'optimizations_per_month': 5,
            'max_model_size_mb': 10,
            'enable_nas': False,
            'priority_queue': False,
            'concurrent_jobs': 1
        },
        'basic': {
            'optimizations_per_month': 50,
            'max_model_size_mb': 100,
            'enable_nas': False,
            'priority_queue': False,
            'concurrent_jobs': 2
        },
        'pro': {
            'optimizations_per_month': 200,
            'max_model_size_mb': 500,
            'enable_nas': True,
            'priority_queue': True,
            'concurrent_jobs': 5
        },
        'enterprise': {
            'optimizations_per_month': -1,  # Unlimited
            'max_model_size_mb': -1,        # Unlimited
            'enable_nas': True,
            'priority_queue': True,
            'concurrent_jobs': 10
        }
    }
    
    # Hardware target specifications
    HARDWARE_TARGETS = {
        'cortex-m4': {
            'name': 'ARM Cortex-M4',
            'cpu_mhz': 84,
            'ram_kb': 256,
            'flash_kb': 1024,
            'power_budget_mw': 15,
            'instruction_set': 'ARM Thumb-2'
        },
        'cortex-m7': {
            'name': 'ARM Cortex-M7',
            'cpu_mhz': 216,
            'ram_kb': 512,
            'flash_kb': 2048,
            'power_budget_mw': 25,
            'instruction_set': 'ARM Thumb-2'
        },
        'risc-v': {
            'name': 'RISC-V MCU',
            'cpu_mhz': 100,
            'ram_kb': 128,
            'flash_kb': 512,
            'power_budget_mw': 12,
            'instruction_set': 'RISC-V RV32I'
        },
        'esp32': {
            'name': 'ESP32',
            'cpu_mhz': 240,
            'ram_kb': 520,
            'flash_kb': 4096,
            'power_budget_mw': 20,
            'instruction_set': 'Xtensa LX6'
        },
        'stm32': {
            'name': 'STM32',
            'cpu_mhz': 168,
            'ram_kb': 384,
            'flash_kb': 1536,
            'power_budget_mw': 18,
            'instruction_set': 'ARM Cortex-M4'
        },
        'arm-cortex-a': {
            'name': 'ARM Cortex-A',
            'cpu_mhz': 1000,
            'ram_kb': 1024,
            'flash_kb': 8192,
            'power_budget_mw': 50,
            'instruction_set': 'ARM A32/A64'
        }
    }

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    WTF_CSRF_ENABLED = False

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False
    
    # Use PostgreSQL in production
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'postgresql://user:password@localhost/tinyml_optimizer'
    
    # Production security settings
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}
