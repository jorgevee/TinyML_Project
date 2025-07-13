from flask import Blueprint, request, jsonify, current_app, send_file
from werkzeug.utils import secure_filename
import os
import uuid
from datetime import datetime
from extensions import db
from models import User, Project, OptimizationJob, BenchmarkResult, Subscription
from models import SubscriptionTier, OptimizationStatus, HardwareTarget, ModelFramework
from optimization_pipeline import optimize_model_task
from processors.upload_processor import process_model_upload
from validators.model_validator import validate_uploaded_model
import json
import logging

logger = logging.getLogger(__name__)

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

@api_bp.route('/validate-model', methods=['POST'])
def validate_model_file():
    """Validate a model file without starting optimization"""
    try:
        # Check if file is present
        if 'model_file' not in request.files:
            return jsonify({'error': 'No model file provided'}), 400
        
        file = request.files['model_file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Get expected framework (optional)
        expected_framework = request.form.get('model_framework')
        
        # Validate the file
        validation_result = validate_uploaded_model(file, expected_framework)
        
        # Add additional metadata
        response_data = {
            'validation_result': validation_result,
            'file_name': file.filename,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # If validation successful, include recommendations
        if validation_result['is_valid']:
            file_info = validation_result.get('file_info', {})
            response_data['recommendations'] = _get_optimization_recommendations(file_info)
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"Model validation error: {str(e)}")
        return jsonify({'error': f'Validation failed: {str(e)}'}), 500

def _get_optimization_recommendations(file_info: dict) -> dict:
    """Generate optimization recommendations based on model analysis"""
    recommendations = {
        'suggested_optimizations': [],
        'hardware_targets': [],
        'estimated_compression': {}
    }
    
    total_params = file_info.get('total_params', 0)
    model_type = file_info.get('model_type', 'unknown')
    
    # Recommend optimizations based on model size
    if total_params > 1_000_000:  # Large model
        recommendations['suggested_optimizations'].extend([
            'quantization', 'pruning', 'nas'
        ])
        recommendations['estimated_compression']['quantization'] = '50-75%'
        recommendations['estimated_compression']['pruning'] = '30-60%'
    elif total_params > 100_000:  # Medium model
        recommendations['suggested_optimizations'].extend([
            'quantization', 'pruning'
        ])
        recommendations['estimated_compression']['quantization'] = '40-65%'
        recommendations['estimated_compression']['pruning'] = '20-50%'
    else:  # Small model
        recommendations['suggested_optimizations'].append('quantization')
        recommendations['estimated_compression']['quantization'] = '30-50%'
    
    # Recommend hardware targets based on model complexity
    if total_params < 50_000:
        recommendations['hardware_targets'] = ['cortex-m4', 'cortex-m7', 'esp32']
    elif total_params < 500_000:
        recommendations['hardware_targets'] = ['cortex-m7', 'esp32', 'arm-cortex-a']
    else:
        recommendations['hardware_targets'] = ['arm-cortex-a', 'cortex-m7']
    
    return recommendations

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
    """Start a new optimization job with enhanced validation and processing"""
    try:
        # Check if file is present
        if 'model_file' not in request.files:
            return jsonify({'error': 'No model file provided'}), 400
        
        file = request.files['model_file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Get form data
        model_framework = request.form.get('model_framework')
        hardware_target = request.form.get('hardware_target')
        
        # Validate required fields
        if not hardware_target or not model_framework:
            return jsonify({'error': 'Hardware target and model framework are required'}), 400
        
        # Validate enum values
        try:
            HardwareTarget(hardware_target)
            ModelFramework(model_framework)
        except ValueError:
            return jsonify({'error': 'Invalid hardware target or model framework'}), 400
        
        # Collect optimization parameters
        optimization_params = {
            'hardware_target': hardware_target,
            'target_size_kb': request.form.get('target_size_kb', 100, type=int),
            'target_accuracy_threshold': request.form.get('target_accuracy_threshold', 0.95, type=float),
            'enable_quantization': request.form.get('enable_quantization', 'true').lower() == 'true',
            'enable_pruning': request.form.get('enable_pruning', 'true').lower() == 'true',
            'enable_nas': request.form.get('enable_nas', 'false').lower() == 'true',
            'quantization_type': request.form.get('quantization_type', 'int8'),
            'pruning_sparsity': request.form.get('pruning_sparsity', 0.5, type=float)
        }
        
        # Process upload with enhanced validation
        success, result = process_model_upload(
            file=file,
            project_id=project_id,
            model_framework=model_framework,
            **optimization_params
        )
        
        if not success:
            logger.error(f"Upload processing failed: {result}")
            return jsonify(result), 400
        
        # Start background optimization task
        job_id = result['job_id']
        optimize_model_task.delay(job_id)
        
        logger.info(f"Started optimization job {job_id} for project {project_id}")
        
        # Return success response with job details
        response_data = {
            'job_id': job_id,
            'optimization_job': result['optimization_job'],
            'file_info': result['file_info'],
            'message': 'Optimization job started successfully'
        }
        
        if result.get('warnings'):
            response_data['warnings'] = result['warnings']
        
        return jsonify(response_data), 201
        
    except Exception as e:
        logger.error(f"Optimization start error: {str(e)}")
        return jsonify({'error': f'Failed to start optimization: {str(e)}'}), 500

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
    
    # Check if file exists
    if not os.path.exists(job.optimized_model_path):
        return jsonify({'error': 'Optimized model file not found on disk'}), 404
    
    try:
        # Return the actual file for download
        return send_file(
            job.optimized_model_path,
            as_attachment=True,
            download_name=f"optimized_{os.path.basename(job.optimized_model_path)}",
            mimetype='application/octet-stream'
        )
    except Exception as e:
        logger.error(f"Download error for job {job_id}: {str(e)}")
        return jsonify({'error': 'Failed to download file'}), 500

@api_bp.route('/jobs/<job_id>/files/<file_type>', methods=['GET'])
def get_job_file_info(job_id, file_type):
    """Get information about job files"""
    job = OptimizationJob.query.filter_by(job_id=job_id).first_or_404()
    
    from processors.upload_processor import UploadProcessor
    processor = UploadProcessor()
    file_paths = processor.get_job_file_paths(job_id)
    
    if file_type == 'original':
        file_path = job.original_model_path
    elif file_type == 'optimized_model':
        file_path = job.optimized_model_path
    else:
        return jsonify({'error': 'Invalid file type'}), 400
    
    if not file_path or not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404
    
    file_stat = os.stat(file_path)
    
    return jsonify({
        'file_path': file_path,
        'file_size': file_stat.st_size,
        'created_at': datetime.fromtimestamp(file_stat.st_ctime).isoformat(),
        'modified_at': datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
        'download_url': f'/api/jobs/{job_id}/download' if file_type == 'optimized_model' else None
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
