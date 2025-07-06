"""Base optimizer interface for different ML frameworks"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class BaseOptimizer(ABC):
    """Abstract base class for framework-specific optimizers"""
    
    def __init__(self, model_path: str, config: Dict[str, Any]):
        self.model_path = model_path
        self.config = config
        self.model_info = {}
        
    @abstractmethod
    def load_model(self) -> Any:
        """Load the model from file"""
        pass
    
    @abstractmethod
    def analyze_model(self) -> Dict[str, Any]:
        """Analyze model structure and return metadata"""
        pass
    
    @abstractmethod
    def apply_quantization(self, quantization_type: str) -> Tuple[Any, float]:
        """Apply quantization and return (model, size_reduction_factor)"""
        pass
    
    @abstractmethod
    def apply_pruning(self, sparsity: float) -> Tuple[Any, float]:
        """Apply pruning and return (model, size_reduction_factor)"""
        pass
    
    @abstractmethod
    def save_optimized_model(self, model: Any, output_path: str) -> None:
        """Save the optimized model to file"""
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return self.model_info
    
    def log_optimization_step(self, step: str, details: Dict[str, Any]) -> None:
        """Log optimization step details"""
        logger.info(f"Optimization step '{step}': {details}")
