from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
import os
import uuid
from datetime import datetime
from extensions import db
from models import User, Project, OptimizationJob, BenchmarkResult, Subscription
from models import SubscriptionTier, OptimizationStatus, HardwareTarget, ModelFramework
from optimization_pipeline import optimize_model_task
import json

api_bp = Blueprint('api', __name__)

# Allowed file extensions for model uploads
ALLOWED_EXTENSIONS = {
    'h5', 'pb', 'tflite', 'pth', 'pt', 'onnx', 'pkl', 'joblib'
}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@api_bp.route('/docs')
def api_docs():
    """API documentation endpoint"""
    return jsonify({
        'title': 'TinyML Model Optimization Pipeline API',
        'version': '1.0.0',
        'endpoints': {
            'POST /api/projects': 'Create a new project',
            'GET /api/projects': 'List user projects',
            'GET /api/projects/<id>': 'Get project details',
            'POST /api/projects/<id>/optimize': 'Start optimization job',
            'GET /api/jobs/<job_id>': 'Get optimization job status',
            'GET /api/jobs/<job_id>/results': 'Get optimization results',
            'GET /api/hardware-targets': 'List supported hardware targets',
            'GET /api/subscription': 'Get user subscription info'
        }
    })

@api_bp.route('/hardware-targets', methods=['GET'])
def get_hardware_targets():
    """Get list of supported hardware targets"""
    targets = [
        {
            'value': target.value,
            'name': target.value.replace('-', ' ').title(),
            'description': f'{target.value.upper()} microcontroller'
        }
        for target in HardwareTarget
    ]
    return jsonify({'hardware_targets': targets})

@api_bp.route('/model-frameworks', methods=['GET'])
def get_model_frameworks():
    """Get list of supported model frameworks"""
    frameworks = [
        {
            'value': framework.value,
            'name': framework.value.title(),
            'extensions': {
                'tensorflow': ['.h5', '.pb', '.tflite'],
                'pytorch': ['.pth', '.pt'],
                'onnx': ['.onnx'],
                'tflite': ['.tflite']
            }.get(framework.value, [])
        }
        for framework in ModelFramework
    ]
    return jsonify({'model_frameworks': frameworks})

@api_bp.route('/projects', methods=['POST'])
def create_project():
    """Create a new project"""
    data = request.get_json()
    
    if not data or 'name' not in data:
        return jsonify({'error': 'Project name is required'}), 400
    
    # For demo purposes, using a default user_id
    # In production, this would come from authentication
    user_id = data.get('user_id', 1)
    
    project = Project(
        user_id=user_id,
        name=data['name'],
        description=data.get('description', '')
    )
    
    try:
        db.session.add(project)
        db.session.commit()
        return jsonify(project.to_dict()), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@api_bp.route('/projects', methods=['GET'])
def get_projects():
    """Get list of user projects"""
    # For demo purposes, using a default user_id
    user_id = request.args.get('user_id', 1, type=int)
    
    projects = Project.query.filter_by(user_id=user_id).all()
    return jsonify({
        'projects': [project.to_dict() for project in projects]
    })

@api_bp.route('/projects/<int:project_id>', methods=['GET'])
def get_project(project_id):
    """Get project details"""
    project = Project.query.get_or_404(project_id)
    
    project_data = project.to_dict()
    project_data['optimization_jobs'] = [job.to_dict() for job in project.optimization_jobs]
    
    return jsonify(project_data)

@api_bp.route('/projects/<int:project_id>/optimize', methods=['POST'])
def start_optimization(project_id):
    """Start a new optimization job"""
    project = Project.query.get_or_404(project_id)
    
    # Check if file is present
    if 'model_file' not in request.files:
        return jsonify({'error': 'No model file provided'}), 400
    
    file = request.files['model_file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not supported'}), 400
    
    # Get form data
    hardware_target = request.form.get('hardware_target')
    model_framework = request.form.get('model_framework')
    target_size_kb = request.form.get('target_size_kb', 100, type=int)
    target_accuracy = request.form.get('target_accuracy_threshold', 0.95, type=float)
    
    # Validate required fields
    if not hardware_target or not model_framework:
        return jsonify({'error': 'Hardware target and model framework are required'}), 400
    
    try:
        hardware_target_enum = HardwareTarget(hardware_target)
        model_framework_enum = ModelFramework(model_framework)
    except ValueError:
        return jsonify({'error': 'Invalid hardware target or model framework'}), 400
    
    # Save uploaded file
    filename = secure_filename(file.filename)
    job_id = str(uuid.uuid4())
    upload_path = os.path.join(current_app.config['UPLOAD_FOLDER'], job_id)
    os.makedirs(upload_path, exist_ok=True)
    
    file_path = os.path.join(upload_path, filename)
    file.save(file_path)
    
    # Create optimization job
    optimization_job = OptimizationJob(
        project_id=project_id,
        job_id=job_id,
        original_model_path=file_path,
        model_framework=model_framework_enum,
        hardware_target=hardware_target_enum,
        target_size_kb=target_size_kb,
        target_accuracy_threshold=target_accuracy,
        enable_quantization=request.form.get('enable_quantization', 'true').lower() == 'true',
        enable_pruning=request.form.get('enable_pruning', 'true').lower() == 'true',
        enable_nas=request.form.get('enable_nas', 'false').lower() == 'true',
        quantization_type=request.form.get('quantization_type', 'int8'),
        pruning_sparsity=request.form.get('pruning_sparsity', 0.5, type=float),
        original_size_bytes=os.path.getsize(file_path)
    )
    
    try:
        db.session.add(optimization_job)
        db.session.commit()
        
        # Start background optimization task
        optimize_model_task.delay(job_id)
        
        return jsonify(optimization_job.to_dict()), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@api_bp.route('/jobs/<job_id>', methods=['GET'])
def get_optimization_job(job_id):
    """Get optimization job status and details"""
    job = OptimizationJob.query.filter_by(job_id=job_id).first_or_404()
    return jsonify(job.to_dict())

@api_bp.route('/jobs/<job_id>/results', methods=['GET'])
def get_optimization_results(job_id):
    """Get optimization job results including benchmark data"""
    job = OptimizationJob.query.filter_by(job_id=job_id).first_or_404()
    
    if job.status != OptimizationStatus.COMPLETED:
        return jsonify({'error': 'Optimization not completed yet'}), 400
    
    job_data = job.to_dict()
    job_data['benchmark_results'] = [result.to_dict() for result in job.benchmark_results]
    
    return jsonify(job_data)

@api_bp.route('/jobs/<job_id>/download', methods=['GET'])
def download_optimized_model(job_id):
    """Download the optimized model file"""
    job = OptimizationJob.query.filter_by(job_id=job_id).first_or_404()
    
    if job.status != OptimizationStatus.COMPLETED or not job.optimized_model_path:
        return jsonify({'error': 'Optimized model not available'}), 400
    
    # In a real implementation, this would return the file
    # For now, return the file path
    return jsonify({
        'download_url': f'/api/jobs/{job_id}/files/optimized_model',
        'file_path': job.optimized_model_path,
        'file_size_bytes': job.optimized_size_bytes
    })

@api_bp.route('/subscription', methods=['GET'])
def get_subscription():
    """Get user subscription information"""
    # For demo purposes, using a default user_id
    user_id = request.args.get('user_id', 1, type=int)
    
    subscription = Subscription.query.filter_by(user_id=user_id).first()
    
    if not subscription:
        # Create default free subscription
        subscription = Subscription(
            user_id=user_id,
            tier=SubscriptionTier.FREE,
            optimizations_limit=5,
            max_model_size_mb=10
        )
        db.session.add(subscription)
        db.session.commit()
    
    return jsonify(subscription.to_dict())

@api_bp.route('/subscription/upgrade', methods=['POST'])
def upgrade_subscription():
    """Upgrade user subscription"""
    data = request.get_json()
    user_id = data.get('user_id', 1)
    new_tier = data.get('tier')
    
    if not new_tier:
        return jsonify({'error': 'Subscription tier is required'}), 400
    
    try:
        tier_enum = SubscriptionTier(new_tier)
    except ValueError:
        return jsonify({'error': 'Invalid subscription tier'}), 400
    
    subscription = Subscription.query.filter_by(user_id=user_id).first()
    
    if not subscription:
        return jsonify({'error': 'Subscription not found'}), 404
    
    # Update subscription limits based on tier
    tier_limits = {
        SubscriptionTier.FREE: {'limit': 5, 'size': 10},
        SubscriptionTier.BASIC: {'limit': 50, 'size': 100},
        SubscriptionTier.PRO: {'limit': 200, 'size': 500},
        SubscriptionTier.ENTERPRISE: {'limit': -1, 'size': -1}  # Unlimited
    }
    
    limits = tier_limits.get(tier_enum, tier_limits[SubscriptionTier.FREE])
    
    subscription.tier = tier_enum
    subscription.optimizations_limit = limits['limit']
    subscription.max_model_size_mb = limits['size']
    
    try:
        db.session.commit()
        return jsonify(subscription.to_dict())
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@api_bp.route('/stats', methods=['GET'])
def get_platform_stats():
    """Get platform statistics"""
    total_jobs = OptimizationJob.query.count()
    completed_jobs = OptimizationJob.query.filter_by(status=OptimizationStatus.COMPLETED).count()
    total_users = User.query.count()
    
    # Calculate average compression ratio
    completed_job_results = OptimizationJob.query.filter(
        OptimizationJob.status == OptimizationStatus.COMPLETED,
        OptimizationJob.compression_ratio.isnot(None)
    ).all()
    
    avg_compression = 0
    if completed_job_results:
        avg_compression = sum(job.compression_ratio for job in completed_job_results) / len(completed_job_results)
    
    return jsonify({
        'total_optimizations': total_jobs,
        'completed_optimizations': completed_jobs,
        'total_users': total_users,
        'average_compression_ratio': round(avg_compression, 2),
        'success_rate': round((completed_jobs / total_jobs * 100) if total_jobs > 0 else 0, 1)
    })
