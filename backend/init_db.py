#!/usr/bin/env python3
"""
Database initialization script for TinyML Optimization Pipeline

This script creates the database tables and optionally seeds with sample data.
Run with: python init_db.py
"""

import os
import sys
from datetime import datetime, timedelta
from werkzeug.security import generate_password_hash

# Add the backend directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app, db
from models import (
    User, Project, OptimizationJob, BenchmarkResult, Subscription,
    SubscriptionTier, OptimizationStatus, HardwareTarget, ModelFramework
)

def create_tables():
    """Create all database tables"""
    print("Creating database tables...")
    with app.app_context():
        db.create_all()
        print("✅ Database tables created successfully!")

def seed_sample_data():
    """Seed database with sample data for testing"""
    print("Seeding sample data...")
    
    with app.app_context():
        # Create sample users
        sample_users = [
            {
                'username': 'demo_user',
                'email': 'demo@tinyml.com',
                'password': 'demo123'
            },
            {
                'username': 'researcher',
                'email': 'researcher@university.edu',
                'password': 'research123'
            },
            {
                'username': 'enterprise',
                'email': 'admin@company.com',
                'password': 'enterprise123'
            }
        ]
        
        users = []
        for user_data in sample_users:
            # Check if user already exists
            existing_user = User.query.filter_by(email=user_data['email']).first()
            if not existing_user:
                user = User(
                    username=user_data['username'],
                    email=user_data['email'],
                    password_hash=generate_password_hash(user_data['password'])
                )
                db.session.add(user)
                users.append(user)
                print(f"  Created user: {user_data['username']}")
            else:
                users.append(existing_user)
                print(f"  User already exists: {user_data['username']}")
        
        db.session.commit()
        
        # Create subscriptions for users
        subscription_data = [
            {'tier': SubscriptionTier.FREE, 'limit': 5, 'size': 10},
            {'tier': SubscriptionTier.PRO, 'limit': 200, 'size': 500},
            {'tier': SubscriptionTier.ENTERPRISE, 'limit': -1, 'size': -1}
        ]
        
        for i, user in enumerate(users):
            if not user.subscription:
                tier_data = subscription_data[i % len(subscription_data)]
                subscription = Subscription(
                    user_id=user.id,
                    tier=tier_data['tier'],
                    optimizations_limit=tier_data['limit'],
                    max_model_size_mb=tier_data['size'],
                    expires_at=datetime.utcnow() + timedelta(days=365)
                )
                db.session.add(subscription)
                print(f"  Created {tier_data['tier'].value} subscription for {user.username}")
        
        db.session.commit()
        
        # Create sample projects
        sample_projects = [
            {
                'name': 'Image Classification',
                'description': 'CNN model for image classification on Cortex-M4',
                'user_id': users[0].id
            },
            {
                'name': 'Voice Recognition',
                'description': 'RNN model for keyword spotting on ESP32',
                'user_id': users[0].id
            },
            {
                'name': 'Anomaly Detection',
                'description': 'Autoencoder for industrial sensor monitoring',
                'user_id': users[1].id
            },
            {
                'name': 'Object Detection',
                'description': 'YOLO-tiny for real-time object detection',
                'user_id': users[2].id
            }
        ]
        
        projects = []
        for proj_data in sample_projects:
            project = Project(
                name=proj_data['name'],
                description=proj_data['description'],
                user_id=proj_data['user_id']
            )
            db.session.add(project)
            projects.append(project)
            print(f"  Created project: {proj_data['name']}")
        
        db.session.commit()
        
        # Create sample optimization jobs
        import uuid
        sample_jobs = [
            {
                'project_id': projects[0].id,
                'model_framework': ModelFramework.TENSORFLOW,
                'hardware_target': HardwareTarget.CORTEX_M4,
                'status': OptimizationStatus.COMPLETED,
                'original_size': 2_500_000,
                'optimized_size': 85_000,
                'accuracy_before': 0.92,
                'accuracy_after': 0.89
            },
            {
                'project_id': projects[1].id,
                'model_framework': ModelFramework.PYTORCH,
                'hardware_target': HardwareTarget.ESP32,
                'status': OptimizationStatus.PROCESSING,
                'original_size': 1_800_000,
                'optimized_size': None,
                'accuracy_before': 0.94,
                'accuracy_after': None
            },
            {
                'project_id': projects[2].id,
                'model_framework': ModelFramework.ONNX,
                'hardware_target': HardwareTarget.RISC_V,
                'status': OptimizationStatus.COMPLETED,
                'original_size': 3_200_000,
                'optimized_size': 95_000,
                'accuracy_before': 0.88,
                'accuracy_after': 0.85
            }
        ]
        
        for job_data in sample_jobs:
            job = OptimizationJob(
                project_id=job_data['project_id'],
                job_id=str(uuid.uuid4()),
                original_model_path=f"/uploads/sample_model_{job_data['project_id']}.h5",
                model_framework=job_data['model_framework'],
                hardware_target=job_data['hardware_target'],
                status=job_data['status'],
                original_size_bytes=job_data['original_size'],
                optimized_size_bytes=job_data['optimized_size'],
                accuracy_before=job_data['accuracy_before'],
                accuracy_after=job_data['accuracy_after'],
                progress_percentage=100 if job_data['status'] == OptimizationStatus.COMPLETED else 45,
                started_at=datetime.utcnow() - timedelta(hours=2),
                completed_at=datetime.utcnow() - timedelta(hours=1) if job_data['status'] == OptimizationStatus.COMPLETED else None
            )
            
            if job_data['optimized_size']:
                job.compression_ratio = job_data['original_size'] / job_data['optimized_size']
                job.optimized_model_path = f"/uploads/optimized_model_{job_data['project_id']}.tflite"
            
            db.session.add(job)
            print(f"  Created optimization job for project {job_data['project_id']}")
            
            # Add benchmark results for completed jobs
            if job_data['status'] == OptimizationStatus.COMPLETED:
                benchmark = BenchmarkResult(
                    optimization_job_id=job.id,
                    flash_usage_bytes=job_data['optimized_size'] + 20_000,
                    ram_usage_bytes=45_000,
                    inference_time_ms=12.5,
                    cycles_per_inference=1_050_000,
                    power_consumption_mw=18.2,
                    energy_per_inference_uj=227.5,
                    throughput_inferences_per_second=80.0,
                    latency_percentile_95_ms=15.2
                )
                db.session.add(benchmark)
                print(f"    Added benchmark results")
        
        db.session.commit()
        print("✅ Sample data seeded successfully!")

def reset_database():
    """Drop all tables and recreate them"""
    print("⚠️  Resetting database (this will delete all data)...")
    response = input("Are you sure you want to continue? (y/N): ")
    
    if response.lower() != 'y':
        print("Database reset cancelled.")
        return
    
    with app.app_context():
        db.drop_all()
        print("  Dropped all tables")
        db.create_all()
        print("  Recreated all tables")
        print("✅ Database reset completed!")

def main():
    """Main function"""
    print("TinyML Optimization Pipeline - Database Setup")
    print("=" * 50)
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "reset":
            reset_database()
            return
        elif command == "seed":
            seed_sample_data()
            return
        elif command == "create":
            create_tables()
            return
        else:
            print(f"Unknown command: {command}")
            print("Available commands: create, seed, reset")
            return
    
    # Default: create tables and ask about seeding
    create_tables()
    
    seed_response = input("\nWould you like to seed the database with sample data? (y/N): ")
    if seed_response.lower() == 'y':
        seed_sample_data()
    
    print("\n🎉 Database setup completed!")
    print("\nNext steps:")
    print("1. Start Redis: redis-server")
    print("2. Start Celery worker: celery -A celery_worker.celery worker --loglevel=info")
    print("3. Start Flask app: python app.py")
    print("4. Visit http://localhost:5000 to test the API")

if __name__ == "__main__":
    main()
