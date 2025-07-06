"""Optimizers package for different ML frameworks"""

from .base import BaseOptimizer
from .tensorflow import TensorFlowOptimizer
from .pytorch import PyTorchOptimizer

# Optimizer factory function
def get_optimizer(framework: str, model_path: str, config: dict) -> BaseOptimizer:
    """Factory function to get the appropriate optimizer for a given framework"""
    
    framework_map = {
        'tensorflow': TensorFlowOptimizer,
        'pytorch': PyTorchOptimizer,
    }
    
    if framework not in framework_map:
        raise ValueError(f"Unsupported framework: {framework}. Supported frameworks: {list(framework_map.keys())}")
    
    optimizer_class = framework_map[framework]
    return optimizer_class(model_path, config)

__all__ = ['BaseOptimizer', 'TensorFlowOptimizer', 'PyTorchOptimizer', 'get_optimizer']
