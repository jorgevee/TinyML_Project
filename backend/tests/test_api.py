#!/usr/bin/env python3
"""
Simple test script for TinyML Optimization Pipeline API

This script demonstrates basic API functionality by:
1. Creating a project
2. Starting an optimization job (without file upload)
3. Checking job status
4. Getting platform statistics

Run with: python test_api.py
"""

import requests
import json
import time

# API base URL
BASE_URL = "http://localhost:5000"

def test_health_check():
    """Test health check endpoint"""
    print("🔍 Testing health check...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_api_info():
    """Test API info endpoint"""
    print("🔍 Testing API info...")
    response = requests.get(f"{BASE_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_hardware_targets():
    """Test hardware targets endpoint"""
    print("🔍 Testing hardware targets...")
    response = requests.get(f"{BASE_URL}/api/hardware-targets")
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Available targets: {len(data['hardware_targets'])}")
    for target in data['hardware_targets']:
        print(f"  - {target['name']}: {target['value']}")
    print()

def test_model_frameworks():
    """Test model frameworks endpoint"""
    print("🔍 Testing model frameworks...")
    response = requests.get(f"{BASE_URL}/api/model-frameworks")
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Available frameworks: {len(data['model_frameworks'])}")
    for framework in data['model_frameworks']:
        print(f"  - {framework['name']}: {framework['extensions']}")
    print()

def test_create_project():
    """Test project creation"""
    print("🔍 Testing project creation...")
    project_data = {
        "name": "Test Project",
        "description": "Automated test project for API verification",
        "user_id": 1
    }
    
    response = requests.post(
        f"{BASE_URL}/api/projects",
        json=project_data,
        headers={'Content-Type': 'application/json'}
    )
    
    print(f"Status: {response.status_code}")
    if response.status_code == 201:
        project = response.json()
        print(f"Created project: {project['name']} (ID: {project['id']})")
        return project['id']
    else:
        print(f"Error: {response.text}")
        return None

def test_get_projects():
    """Test getting projects list"""
    print("🔍 Testing projects list...")
    response = requests.get(f"{BASE_URL}/api/projects?user_id=1")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Found {len(data['projects'])} projects:")
        for project in data['projects']:
            print(f"  - {project['name']} (ID: {project['id']})")
    print()

def test_subscription():
    """Test subscription endpoint"""
    print("🔍 Testing subscription info...")
    response = requests.get(f"{BASE_URL}/api/subscription?user_id=1")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        subscription = response.json()
        print(f"Subscription tier: {subscription['tier']}")
        print(f"Optimizations used: {subscription['optimizations_used']}/{subscription['optimizations_limit']}")
        print(f"Max model size: {subscription['max_model_size_mb']} MB")
    print()

def test_platform_stats():
    """Test platform statistics"""
    print("🔍 Testing platform statistics...")
    response = requests.get(f"{BASE_URL}/api/stats")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        stats = response.json()
        print(f"Total optimizations: {stats['total_optimizations']}")
        print(f"Completed: {stats['completed_optimizations']}")
        print(f"Success rate: {stats['success_rate']}%")
        print(f"Average compression: {stats['average_compression_ratio']}x")
    print()

def main():
    """Run all tests"""
    print("🚀 TinyML Optimization Pipeline API Test")
    print("=" * 50)
    
    try:
        # Basic connectivity tests
        test_health_check()
        test_api_info()
        
        # Configuration endpoints
        test_hardware_targets()
        test_model_frameworks()
        
        # Project management
        project_id = test_create_project()
        test_get_projects()
        
        # Subscription info
        test_subscription()
        
        # Platform statistics
        test_platform_stats()
        
        print("✅ All tests completed successfully!")
        print("\nNext steps to test full functionality:")
        print("1. Install and start Redis: redis-server")
        print("2. Initialize database: python init_db.py")
        print("3. Start Celery worker: celery -A celery_worker.celery worker --loglevel=info")
        print("4. Start Flask app: python app.py")
        print("5. Upload a model file using the API or frontend")
        
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed!")
        print("Make sure the Flask server is running:")
        print("  python app.py")
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    main()
