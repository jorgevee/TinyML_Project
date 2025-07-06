#!/usr/bin/env python3
"""
Celery worker for TinyML Model Optimization Pipeline

This script starts a Celery worker that processes background optimization tasks.
Run with: celery -A celery_worker.celery worker --loglevel=info
"""

import os
from app import app
from extensions import celery

# Import tasks to register them
from optimization_pipeline import optimize_model_task

if __name__ == '__main__':
    # Start the worker
    celery.start()
