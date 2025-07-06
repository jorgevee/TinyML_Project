"""Main optimization pipeline orchestrator"""

import os
import time
from datetime import datetime
from typing import Dict, Any, Optional
from ..core.extensions import db, celery
from ..core.models import OptimizationJob, BenchmarkResult, OptimizationStatus
from ..optimizers import get_optimizer
from .benchmarks import BenchmarkRunner
from .quantization import QuantizationEngine
from .pruning import PruningEngine
from .nas import NASEngine
import logging

logger = logging.getLogger(__name__)

class ModelOptimizer:
    """Main optimization pipeline orchestrator"""
    
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.job: Optional[OptimizationJob] = None
        self.optimizer = None
        self.model = None
        self.model_info = {}
        
    def load_job(self) -> OptimizationJob:
        """Load optimization job from database"""
        self.job = OptimizationJob.query.filter_by(job_id=self.job_id).first()
        if not self.job:
            raise ValueError(f"Job {self.job_id} not found")
        return self.job
    
    def update_progress(self, percentage: int, status: Optional[OptimizationStatus] = None) -> None:
        """Update job progress in database"""
        if self.job:
            self.job.progress_percentage = percentage
            if status:
                self.job.status = status
            db.session.commit()
            logger.info(f"Job {self.job_id}: Progress {percentage}%")
    
    def update_status(self, status: OptimizationStatus, error_message: Optional[str] = None) -> None:
        """Update job status"""
        if self.job:
            self.job.status = status
            if error_message:
                self.job.error_message = error_message
            if status == OptimizationStatus.PROCESSING and not self.job.started_at:
                self.job.started_at = datetime.utcnow()
            elif status == OptimizationStatus.COMPLETED:
                self.job.completed_at = datetime.utcnow()
            db.session.commit()
    
    def initialize_optimizer(self) -> None:
        """Initialize the framework-specific optimizer"""
        if not self.job:
            raise ValueError("Job not loaded")
        
        config = {
            'quantization_enabled': self.job.enable_quantization,
            'quantization_type': self.job.quantization_type,
            'pruning_enabled': self.job.enable_pruning,
            'pruning_sparsity': self.job.pruning_sparsity,
            'nas_enabled': self.job.enable_nas,
            'hardware_target': self.job.hardware_target.value,
            'target_size_kb': self.job.target_size_kb,
            'target_accuracy_threshold': self.job.target_accuracy_threshold
        }
        
        framework = self.job.model_framework.value
        self.optimizer = get_optimizer(framework, self.job.original_model_path, config)
        logger.info(f"Initialized {framework} optimizer for job {self.job_id}")
    
    def analyze_model(self) -> Dict[str, Any]:
        """Analyze the original model"""
        self.update_progress(10, OptimizationStatus.PROCESSING)
        logger.info(f"Analyzing model for job {self.job_id}")
        
        if not self.optimizer:
            self.initialize_optimizer()
        
        # Load and analyze the model
        self.model = self.optimizer.load_model()
        self.model_info = self.optimizer.analyze_model()
        
        logger.info(f"Model analysis complete: {self.model_info}")
        return self.model_info
    
    def apply_quantization(self) -> float:
        """Apply quantization optimization"""
        if not self.job.enable_quantization:
            logger.info("Quantization disabled, skipping")
            return 1.0
        
        self.update_progress(25)
        logger.info(f"Applying {self.job.quantization_type} quantization")
        
        quantization_engine = QuantizationEngine(self.optimizer, self.job.quantization_type)
        self.model, size_reduction = quantization_engine.apply_quantization(self.model)
        
        # Update model info with quantization results
        if 'estimated_accuracy' in self.model_info:
            accuracy_impact = quantization_engine.get_accuracy_impact()
            self.model_info['estimated_accuracy'] *= accuracy_impact
        
        logger.info(f"Quantization complete. Size reduction: {size_reduction}")
        return size_reduction
    
    def apply_pruning(self) -> float:
        """Apply pruning optimization"""
        if not self.job.enable_pruning:
            logger.info("Pruning disabled, skipping")
            return 1.0
        
        self.update_progress(45)
        logger.info(f"Applying pruning with {self.job.pruning_sparsity} sparsity")
        
        pruning_engine = PruningEngine(self.optimizer, self.job.pruning_sparsity)
        self.model, size_reduction = pruning_engine.apply_pruning(self.model)
        
        # Update model info with pruning results
        if 'parameters' in self.model_info:
            self.model_info['parameters'] = int(
                self.model_info['parameters'] * (1 - self.job.pruning_sparsity)
            )
        if 'estimated_accuracy' in self.model_info:
            accuracy_impact = pruning_engine.get_accuracy_impact()
            self.model_info['estimated_accuracy'] *= accuracy_impact
        
        logger.info(f"Pruning complete. Size reduction: {size_reduction}")
        return size_reduction
    
    def apply_nas(self) -> float:
        """Apply Neural Architecture Search optimization"""
        if not self.job.enable_nas:
            logger.info("NAS disabled, skipping")
            return 1.0
        
        self.update_progress(65)
        logger.info(f"Running NAS for {self.job.hardware_target.value}")
        
        nas_engine = NASEngine(
            self.optimizer, 
            self.job.hardware_target.value,
            self.job.target_size_kb
        )
        self.model, size_reduction = nas_engine.optimize_architecture(self.model, self.model_info)
        
        # Update model info with NAS results
        if 'layers' in self.model_info:
            self.model_info['layers'] = int(self.model_info['layers'] * 0.8)
        if 'estimated_accuracy' in self.model_info:
            accuracy_impact = nas_engine.get_accuracy_impact()
            self.model_info['estimated_accuracy'] *= accuracy_impact
        
        logger.info(f"NAS complete. Size reduction: {size_reduction}")
        return size_reduction
    
    def generate_optimized_model(self, total_size_reduction: float) -> None:
        """Generate and save the final optimized model"""
        self.update_progress(80)
        logger.info("Generating optimized model")
        
        # Calculate final model size
        original_size = self.job.original_size_bytes
        optimized_size = int(original_size * total_size_reduction)
        
        # Update job with optimization results
        self.job.optimized_size_bytes = optimized_size
        self.job.accuracy_before = 0.92  # This would come from actual model evaluation
        self.job.accuracy_after = self.model_info.get('estimated_accuracy', 0.85)
        self.job.compression_ratio = original_size / optimized_size if optimized_size > 0 else 1
        
        # Generate optimized model file path
        job_dir = os.path.dirname(self.job.original_model_path)
        optimized_filename = f"optimized_{os.path.basename(self.job.original_model_path)}"
        self.job.optimized_model_path = os.path.join(job_dir, optimized_filename)
        
        # Save the optimized model
        self.optimizer.save_optimized_model(self.model, self.job.optimized_model_path)
        
        db.session.commit()
        logger.info(f"Optimized model saved to {self.job.optimized_model_path}")
    
    def run_benchmarks(self) -> BenchmarkResult:
        """Run performance benchmarks on the optimized model"""
        self.update_progress(90)
        logger.info("Running hardware benchmarks")
        
        benchmark_runner = BenchmarkRunner(
            self.job.hardware_target.value,
            self.model_info,
            self.job.optimized_size_bytes
        )
        
        benchmark_result = benchmark_runner.run_benchmarks(self.job.id)
        
        logger.info("Benchmarks complete")
        return benchmark_result
    
    def run_optimization(self) -> bool:
        """Run the complete optimization pipeline"""
        try:
            self.load_job()
            logger.info(f"Starting optimization pipeline for job {self.job_id}")
            
            # Step 1: Analyze model
            self.analyze_model()
            
            # Step 2: Apply optimizations
            total_size_reduction = 1.0
            
            # Quantization
            quant_reduction = self.apply_quantization()
            total_size_reduction *= quant_reduction
            
            # Pruning
            prune_reduction = self.apply_pruning()
            total_size_reduction *= prune_reduction
            
            # Neural Architecture Search
            nas_reduction = self.apply_nas()
            total_size_reduction *= nas_reduction
            
            # Step 3: Generate optimized model
            self.generate_optimized_model(total_size_reduction)
            
            # Step 4: Run benchmarks
            self.run_benchmarks()
            
            # Complete the job
            self.update_progress(100, OptimizationStatus.COMPLETED)
            logger.info(f"Optimization pipeline completed successfully for job {self.job_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Optimization pipeline failed for job {self.job_id}: {str(e)}")
            self.update_status(OptimizationStatus.FAILED, str(e))
            return False

@celery.task(bind=True)
def optimize_model_task(self, job_id: str):
    """Celery task for model optimization"""
    try:
        optimizer = ModelOptimizer(job_id)
        return optimizer.run_optimization()
    except Exception as e:
        logger.error(f"Optimization task failed for job {job_id}: {str(e)}")
        raise
