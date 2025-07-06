"""Quantization optimization engine"""

# import hashlib
import pickle
from typing import Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class QuantizationEngine:
    """Engine for applying quantization optimizations"""

    DEFAULT_MAX_CACHE_SIZE = 10  # Default cache size limit

    def __init__(self, optimizer, quantization_type: str, max_cache_size: int = None):
        self.optimizer = optimizer
        self.quantization_type = quantization_type
        self.accuracy_impact = 1.0
        self._cache = {}  # Cache for quantization results
        self._max_cache_size = max_cache_size if max_cache_size is not None else self.DEFAULT_MAX_CACHE_SIZE
        
    def apply_quantization(self, model: Any) -> Tuple[Any, float]:
        """Apply quantization to the model with caching"""
        cache_key = self._generate_cache_key(model, self.quantization_type)
        
        # Check if result is already cached
        if cache_key in self._cache:
            logger.info(f"Using cached result for {self.quantization_type} quantization")
            cached_result = self._cache[cache_key]
            
            # Store accuracy impact for later retrieval (from cache)
            self.accuracy_impact = cached_result['accuracy_impact']
            
            return cached_result['quantized_model'], cached_result['size_reduction']
        
        logger.info(f"Applying {self.quantization_type} quantization (not cached)")
        
        # Use the framework-specific optimizer to apply quantization
        quantized_model, size_reduction = self.optimizer.apply_quantization(self.quantization_type, model)
        
        # Store accuracy impact for later retrieval
        self._calculate_accuracy_impact()
        
        # Cache the result
        self._cache_result(cache_key, quantized_model, size_reduction, self.accuracy_impact)
        
        return quantized_model, size_reduction
    
    def _calculate_accuracy_impact(self) -> None:
        """Calculate the accuracy impact of quantization"""
        if self.quantization_type == 'int8':
            self.accuracy_impact = 0.98  # 2% accuracy loss
        elif self.quantization_type == 'fp16':
            self.accuracy_impact = 0.995  # 0.5% accuracy loss
        else:  # mixed precision
            self.accuracy_impact = 0.99  # 1% accuracy loss
    
    def get_accuracy_impact(self) -> float:
        """Get the accuracy impact factor"""
        return self.accuracy_impact
    
    def get_quantization_details(self) -> Dict[str, Any]:
        """Get detailed information about the quantization process"""
        return {
            'type': self.quantization_type,
            'accuracy_impact': self.accuracy_impact,
            'description': self._get_quantization_description()
        }
    
    def _get_quantization_description(self) -> str:
        """Get a description of the quantization type"""
        descriptions = {
            'int8': 'Post-training quantization to 8-bit integers. Provides significant size reduction with moderate accuracy loss.',
            'fp16': 'Half-precision floating point quantization. Balanced approach between size reduction and accuracy preservation.',
            'mixed': 'Mixed precision quantization using both INT8 and FP16. Optimizes for both size and accuracy.'
        }
        return descriptions.get(self.quantization_type, 'Unknown quantization type')
    
    def _generate_cache_key(self, model: Any, quantization_type: str) -> str:
        """Generate a unique cache key for the model and quantization type combination"""
        try:
            # Try to create a hash of the model's state
            if hasattr(model, 'state_dict'):
                # PyTorch model
                model_state = pickle.dumps((model.state_dict()))
            elif hasattr(model, 'get_weights'):
                # TensorFlow/Keras model
                model_state = str([w.shape for w in model.get_weights()])
            else:
                # Fallback: use model's string representation and id
                model_state = f"{str(model)}_{id(model)}"
            
            # Create a combined string for hashing
            cache_string = f"{model_state}_{quantization_type}"
            
            # Generate SHA256 hash for consistent cache key
            return hashlib.sha256(cache_string.encode()).hexdigest()
        
        except Exception as e:
            logger.warning(f"Could not generate proper cache key, using fallback: {e}")
            # Fallback to a simpler approach
            return f"{id(model)}_{quantization_type}"
    
    def _cache_result(self, cache_key: str, quantized_model: Any, size_reduction: float, accuracy_impact: float) -> None:
        """Cache the quantization result with memory management"""
        # Check if cache is full and remove oldest entry if needed
        if len(self._cache) >= self._max_cache_size:
            # Remove the first (oldest) entry
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            logger.debug(f"Removed oldest cache entry: {oldest_key}")
        
        # Store the result in cache
        self._cache[cache_key] = {
            'quantized_model': quantized_model,
            'size_reduction': size_reduction,
            'accuracy_impact': accuracy_impact
        }
        
        logger.debug(f"Cached quantization result for key: {cache_key}")
    
    def clear_cache(self) -> None:
        """Clear the quantization cache"""
        self._cache.clear()
        logger.info("Quantization cache cleared")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about the current cache state"""
        return {
            'cache_size': len(self._cache),
            'max_cache_size': self._max_cache_size,
            'cache_keys': list(self._cache.keys())
        }
