###
# PyTorch GPU Backend for fastSuStaIn
# 
# This module provides GPU acceleration for SuStaIn log likelihood computations
# using PyTorch tensors and CUDA operations.
#
# Authors: GPU Migration Team
###

import torch
import numpy as np
import warnings
from typing import Union, Optional, Tuple, Any
import time


class DeviceManager:
    """Manages GPU/CPU device selection and tensor operations."""
    
    def __init__(self, use_gpu: bool = True, device_id: Optional[int] = None):
        """
        Initialize device manager.
        
        Args:
            use_gpu: Whether to attempt GPU usage
            device_id: Specific GPU device ID (None for auto-selection)
        """
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device_id = device_id
        
        if self.use_gpu:
            if device_id is not None:
                self.device = torch.device(f'cuda:{device_id}')
            else:
                self.device = torch.device('cuda')
            self.torch_dtype = torch.float32  # Use float32 for better GPU performance
            print(f"GPU Backend: Using CUDA device {self.device}")
        else:
            self.device = torch.device('cpu')
            self.torch_dtype = torch.float64  # Use float64 for CPU to maintain precision
            if use_gpu:
                warnings.warn("GPU requested but CUDA not available. Falling back to CPU.")
            print("GPU Backend: Using CPU device")
    
    def to_torch(self, np_array: Union[np.ndarray, list, tuple], requires_grad: bool = False) -> torch.Tensor:
        """Convert numpy array to PyTorch tensor on the appropriate device."""
        if np_array is None:
            return None
        
        # Ensure input is a numpy array
        if not isinstance(np_array, np.ndarray):
            np_array = np.asarray(np_array)
        
        tensor = torch.from_numpy(np_array).to(
            device=self.device, 
            dtype=self.torch_dtype,
            non_blocking=True
        )
        
        if requires_grad:
            tensor.requires_grad_(True)
            
        return tensor
    
    def to_numpy(self, torch_tensor: torch.Tensor) -> np.ndarray:
        """Convert PyTorch tensor back to numpy array."""
        if torch_tensor is None:
            return None
            
        return torch_tensor.detach().cpu().numpy()
    
    def empty_cache(self):
        """Clear GPU memory cache if using GPU."""
        if self.use_gpu:
            torch.cuda.empty_cache()
    
    def get_memory_info(self) -> dict:
        """Get memory usage information."""
        if self.use_gpu:
            return {
                'allocated': torch.cuda.memory_allocated(self.device) / 1024**3,  # GB
                'reserved': torch.cuda.memory_reserved(self.device) / 1024**3,   # GB
                'max_allocated': torch.cuda.max_memory_allocated(self.device) / 1024**3  # GB
            }
        else:
            return {'device': 'CPU'}


class TorchSustainBackend:
    """Main backend class for GPU-accelerated SuStaIn computations."""
    
    def __init__(self, use_gpu: bool = True, device_id: Optional[int] = None, 
                 memory_efficient: bool = True):
        """
        Initialize the PyTorch backend.
        
        Args:
            use_gpu: Whether to use GPU acceleration
            device_id: Specific GPU device ID
            memory_efficient: Whether to use memory-efficient operations
        """
        self.device_manager = DeviceManager(use_gpu, device_id)
        self.memory_efficient = memory_efficient
        self.use_gpu = self.device_manager.use_gpu
        
        # Performance tracking
        self.computation_times = {}
        self.memory_usage = {}
        
    def benchmark_operation(self, operation_name: str):
        """Context manager for benchmarking operations."""
        class BenchmarkContext:
            def __init__(self, backend, name):
                self.backend = backend
                self.name = name
                self.start_time = None
                self.start_memory = None
                
            def __enter__(self):
                self.start_time = time.time()
                if self.backend.use_gpu:
                    torch.cuda.synchronize()
                    self.start_memory = self.backend.device_manager.get_memory_info()
                return self
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                if self.backend.use_gpu:
                    torch.cuda.synchronize()
                end_time = time.time()
                end_memory = self.backend.device_manager.get_memory_info()
                
                self.backend.computation_times[self.name] = end_time - self.start_time
                if self.backend.use_gpu:
                    self.backend.memory_usage[self.name] = {
                        'peak_allocated': end_memory['allocated'] - self.start_memory['allocated'],
                        'peak_reserved': end_memory['reserved'] - self.start_memory['reserved']
                    }
        
        return BenchmarkContext(self, operation_name)
    
    def to_torch(self, np_array: Union[np.ndarray, list, tuple], requires_grad: bool = False) -> torch.Tensor:
        """Convert numpy array to PyTorch tensor."""
        return self.device_manager.to_torch(np_array, requires_grad)
    
    def to_numpy(self, torch_tensor: torch.Tensor) -> np.ndarray:
        """Convert PyTorch tensor to numpy array."""
        return self.device_manager.to_numpy(torch_tensor)
    
    def clear_cache(self):
        """Clear GPU memory cache."""
        self.device_manager.empty_cache()
    
    def get_performance_stats(self) -> dict:
        """Get performance statistics."""
        return {
            'computation_times': self.computation_times,
            'memory_usage': self.memory_usage,
            'device_info': self.device_manager.get_memory_info()
        }


class TorchMissingDataHandler:
    """Handles missing data operations efficiently on GPU."""
    
    def __init__(self, backend: TorchSustainBackend):
        self.backend = backend
        self.device = backend.device_manager.device
        self.dtype = backend.device_manager.torch_dtype
    
    def create_missing_data_mask(self, data_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create masks for missing and observed data."""
        missing_mask = torch.isnan(data_tensor)
        observed_mask = ~missing_mask
        return missing_mask, observed_mask
    
    def handle_missing_data_likelihood(self, 
                                     data_tensor: torch.Tensor,
                                     stage_value_tensor: torch.Tensor,
                                     sigmat_tensor: torch.Tensor,
                                     missing_data_prob: torch.Tensor) -> torch.Tensor:
        """
        Efficiently compute likelihood with missing data handling.
        
        Args:
            data_tensor: Input data with potential NaN values
            stage_value_tensor: Stage values for likelihood computation
            sigmat_tensor: Standard deviation tensor
            missing_data_prob: Probability for missing data
            
        Returns:
            Likelihood values with missing data properly handled
        """
        # Create masks
        missing_mask, observed_mask = self.create_missing_data_mask(data_tensor)
        
        # Compute likelihood for observed data
        x_observed = (data_tensor - stage_value_tensor) / sigmat_tensor
        
        # Initialize result with missing data probabilities
        result = torch.log(missing_data_prob)
        
        # Replace with observed data likelihood where data is available
        result = torch.where(observed_mask, x_observed, result)
        
        return result
    
    def vectorized_missing_data_processing(self, 
                                         data_tensor: torch.Tensor,
                                         stage_values: torch.Tensor,
                                         sigmat_tensor: torch.Tensor,
                                         missing_data_prob: torch.Tensor) -> torch.Tensor:
        """
        Vectorized processing of missing data across all stages.
        
        Args:
            data_tensor: Input data (M, B)
            stage_values: Stage values (B, N+1)
            sigmat_tensor: Standard deviations (M, B)
            missing_data_prob: Missing data probabilities (M, B)
            
        Returns:
            Likelihood tensor (M, N+1)
        """
        M, B = data_tensor.shape
        N_plus_1 = stage_values.shape[1]
        
        # Expand dimensions for broadcasting
        data_expanded = data_tensor.unsqueeze(2)  # (M, B, 1)
        stage_expanded = stage_values.unsqueeze(0)  # (1, B, N+1)
        sigmat_expanded = sigmat_tensor.unsqueeze(2)  # (M, B, 1)
        
        # Compute likelihood for all stages at once
        x_all = (data_expanded - stage_expanded) / sigmat_expanded  # (M, B, N+1)
        
        # Handle missing data
        missing_mask = torch.isnan(data_tensor).unsqueeze(2)  # (M, B, 1)
        observed_mask = ~missing_mask
        
        # Initialize with missing data probabilities
        result = torch.log(missing_data_prob).unsqueeze(2)  # (M, B, 1)
        result = result.expand(-1, -1, N_plus_1)  # (M, B, N+1)
        
        # Replace with observed likelihood where data is available
        result = torch.where(observed_mask, x_all, result)
        
        return result


def create_torch_backend(use_gpu: bool = True, device_id: Optional[int] = None) -> TorchSustainBackend:
    """Factory function to create a TorchSustainBackend instance."""
    return TorchSustainBackend(use_gpu=use_gpu, device_id=device_id)


# Utility functions for common operations
def torch_linspace_local(start: torch.Tensor, end: torch.Tensor, num: int, 
                        device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """PyTorch equivalent of the custom linspace_local2 function."""
    if num <= 1:
        return start.unsqueeze(0)
    
    # Create linear interpolation
    steps = torch.linspace(0, 1, num, device=device, dtype=dtype)
    result = start + steps * (end - start)
    return result


def efficient_tensor_tiling(tensor: torch.Tensor, reps: Tuple[int, ...]) -> torch.Tensor:
    """Efficient tensor tiling using PyTorch operations."""
    return tensor.repeat(reps)


def safe_torch_operations(tensor: torch.Tensor, operation: str, **kwargs) -> torch.Tensor:
    """Perform safe tensor operations with error handling."""
    try:
        if operation == 'log':
            return torch.log(tensor + 1e-250)  # Add small epsilon for numerical stability
        elif operation == 'exp':
            return torch.exp(tensor)
        elif operation == 'sum':
            dim = kwargs.get('dim', None)
            return torch.sum(tensor, dim=dim)
        elif operation == 'square':
            return torch.square(tensor)
        else:
            raise ValueError(f"Unknown operation: {operation}")
    except RuntimeError as e:
        if "out of memory" in str(e):
            warnings.warn(f"GPU out of memory during {operation}. Consider reducing batch size.")
            raise
        else:
            raise
