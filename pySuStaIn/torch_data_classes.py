###
# PyTorch-enabled data classes for fastSuStaIn
# 
# This module extends the existing SuStaIn data classes to support PyTorch tensors
# while maintaining compatibility with the original numpy-based implementations.
#
# Authors: GPU Migration Team
###

import torch
import numpy as np
from typing import Optional, Union, Tuple
from .torch_backend import TorchSustainBackend
from .AbstractSustain import AbstractSustainData


class TorchAbstractSustainData(AbstractSustainData):
    """Base class for PyTorch-enabled SuStaIn data classes."""
    
    def __init__(self, backend: Optional[TorchSustainBackend] = None):
        """
        Initialize with optional PyTorch backend.
        
        Args:
            backend: PyTorch backend for GPU operations
        """
        self.backend = backend
        self._torch_tensors = {}
        self._numpy_arrays = {}
        
    def get_torch_tensor(self, key: str) -> Optional[torch.Tensor]:
        """Get PyTorch tensor for a given key."""
        return self._torch_tensors.get(key)
    
    def set_torch_tensor(self, key: str, tensor: torch.Tensor):
        """Set PyTorch tensor for a given key."""
        self._torch_tensors[key] = tensor
    
    def get_numpy_array(self, key: str) -> Optional[np.ndarray]:
        """Get numpy array for a given key."""
        return self._numpy_arrays.get(key)
    
    def set_numpy_array(self, key: str, array: np.ndarray):
        """Set numpy array for a given key."""
        self._numpy_arrays[key] = array
    
    def to_torch(self, key: str) -> torch.Tensor:
        """Convert numpy array to PyTorch tensor."""
        if key not in self._torch_tensors:
            numpy_array = self._numpy_arrays.get(key)
            if numpy_array is not None and self.backend is not None:
                self._torch_tensors[key] = self.backend.to_torch(numpy_array)
        return self._torch_tensors.get(key)
    
    def to_numpy(self, key: str) -> np.ndarray:
        """Convert PyTorch tensor to numpy array."""
        if key not in self._numpy_arrays:
            torch_tensor = self._torch_tensors.get(key)
            if torch_tensor is not None and self.backend is not None:
                self._numpy_arrays[key] = self.backend.to_numpy(torch_tensor)
        return self._numpy_arrays.get(key)
    
    def clear_torch_cache(self):
        """Clear PyTorch tensor cache to free memory."""
        self._torch_tensors.clear()
        if self.backend is not None:
            self.backend.clear_cache()


class TorchZScoreSustainData(TorchAbstractSustainData):
    """PyTorch-enabled version of ZScoreSustainData."""
    
    def __init__(self, data: np.ndarray, numStages: int, 
                 backend: Optional[TorchSustainBackend] = None):
        """
        Initialize with z-score data.
        
        Args:
            data: Z-score data array (M, B) where M=subjects, B=biomarkers
            numStages: Number of stages
            backend: PyTorch backend for GPU operations
        """
        super().__init__(backend)
        self.set_numpy_array('data', data)
        self.__numStages = numStages
        
    def getNumSamples(self) -> int:
        """Get number of samples."""
        data = self.get_numpy_array('data')
        return data.shape[0] if data is not None else 0
    
    def getNumBiomarkers(self) -> int:
        """Get number of biomarkers."""
        data = self.get_numpy_array('data')
        return data.shape[1] if data is not None else 0
    
    def getNumStages(self) -> int:
        """Get number of stages."""
        return self.__numStages
    
    def reindex(self, index: np.ndarray) -> 'TorchZScoreSustainData':
        """Create new data object with reindexed data."""
        data = self.get_numpy_array('data')
        if data is not None:
            reindexed_data = data[index, :]
            return TorchZScoreSustainData(reindexed_data, self.__numStages, self.backend)
        else:
            return TorchZScoreSustainData(np.array([]), self.__numStages, self.backend)
    
    def get_data_torch(self) -> torch.Tensor:
        """Get data as PyTorch tensor."""
        return self.to_torch('data')
    
    def get_data_numpy(self) -> np.ndarray:
        """Get data as numpy array."""
        return self.get_numpy_array('data')


class TorchMixtureSustainData(TorchAbstractSustainData):
    """PyTorch-enabled version of MixtureSustainData."""
    
    def __init__(self, L_yes: np.ndarray, L_no: np.ndarray, numStages: int,
                 backend: Optional[TorchSustainBackend] = None):
        """
        Initialize with mixture model likelihoods.
        
        Args:
            L_yes: Positive class likelihoods (M, B)
            L_no: Negative class likelihoods (M, B)
            numStages: Number of stages
            backend: PyTorch backend for GPU operations
        """
        super().__init__(backend)
        assert L_yes.shape == L_no.shape, "L_yes and L_no must have the same shape"
        
        self.set_numpy_array('L_yes', L_yes)
        self.set_numpy_array('L_no', L_no)
        self.__numStages = numStages
        
    def getNumSamples(self) -> int:
        """Get number of samples."""
        L_yes = self.get_numpy_array('L_yes')
        return L_yes.shape[0] if L_yes is not None else 0
    
    def getNumBiomarkers(self) -> int:
        """Get number of biomarkers."""
        L_yes = self.get_numpy_array('L_yes')
        return L_yes.shape[1] if L_yes is not None else 0
    
    def getNumStages(self) -> int:
        """Get number of stages."""
        return self.__numStages
    
    def reindex(self, index: np.ndarray) -> 'TorchMixtureSustainData':
        """Create new data object with reindexed data."""
        L_yes = self.get_numpy_array('L_yes')
        L_no = self.get_numpy_array('L_no')
        
        if L_yes is not None and L_no is not None:
            reindexed_L_yes = L_yes[index, :]
            reindexed_L_no = L_no[index, :]
            return TorchMixtureSustainData(reindexed_L_yes, reindexed_L_no, 
                                         self.__numStages, self.backend)
        else:
            return TorchMixtureSustainData(np.array([]), np.array([]), 
                                         self.__numStages, self.backend)
    
    def get_L_yes_torch(self) -> torch.Tensor:
        """Get L_yes as PyTorch tensor."""
        return self.to_torch('L_yes')
    
    def get_L_no_torch(self) -> torch.Tensor:
        """Get L_no as PyTorch tensor."""
        return self.to_torch('L_no')
    
    def get_L_yes_numpy(self) -> np.ndarray:
        """Get L_yes as numpy array."""
        return self.get_numpy_array('L_yes')
    
    def get_L_no_numpy(self) -> np.ndarray:
        """Get L_no as numpy array."""
        return self.get_numpy_array('L_no')


class TorchOrdinalSustainData(TorchAbstractSustainData):
    """PyTorch-enabled version of OrdinalSustainData."""
    
    def __init__(self, prob_nl: np.ndarray, prob_score: np.ndarray, numStages: int,
                 backend: Optional[TorchSustainBackend] = None):
        """
        Initialize with ordinal data probabilities.
        
        Args:
            prob_nl: Normal/negative class probabilities (M, B)
            prob_score: Score probabilities (M, B, num_scores)
            numStages: Number of stages
            backend: PyTorch backend for GPU operations
        """
        super().__init__(backend)
        self.set_numpy_array('prob_nl', prob_nl)
        self.set_numpy_array('prob_score', prob_score)
        self.__numStages = numStages
        
    def getNumSamples(self) -> int:
        """Get number of samples."""
        prob_nl = self.get_numpy_array('prob_nl')
        return prob_nl.shape[0] if prob_nl is not None else 0
    
    def getNumBiomarkers(self) -> int:
        """Get number of biomarkers."""
        prob_nl = self.get_numpy_array('prob_nl')
        return prob_nl.shape[1] if prob_nl is not None else 0
    
    def getNumStages(self) -> int:
        """Get number of stages."""
        return self.__numStages
    
    def reindex(self, index: np.ndarray) -> 'TorchOrdinalSustainData':
        """Create new data object with reindexed data."""
        prob_nl = self.get_numpy_array('prob_nl')
        prob_score = self.get_numpy_array('prob_score')
        
        if prob_nl is not None and prob_score is not None:
            reindexed_prob_nl = prob_nl[index, :]
            reindexed_prob_score = prob_score[index, :]
            return TorchOrdinalSustainData(reindexed_prob_nl, reindexed_prob_score,
                                         self.__numStages, self.backend)
        else:
            return TorchOrdinalSustainData(np.array([]), np.array([]),
                                         self.__numStages, self.backend)
    
    def get_prob_nl_torch(self) -> torch.Tensor:
        """Get prob_nl as PyTorch tensor."""
        return self.to_torch('prob_nl')
    
    def get_prob_score_torch(self) -> torch.Tensor:
        """Get prob_score as PyTorch tensor."""
        return self.to_torch('prob_score')
    
    def get_prob_nl_numpy(self) -> np.ndarray:
        """Get prob_nl as numpy array."""
        return self.get_numpy_array('prob_nl')
    
    def get_prob_score_numpy(self) -> np.ndarray:
        """Get prob_score as numpy array."""
        return self.get_numpy_array('prob_score')


# Factory functions for creating PyTorch-enabled data objects
def create_torch_zscore_data(data: np.ndarray, numStages: int, 
                           backend: Optional[TorchSustainBackend] = None) -> TorchZScoreSustainData:
    """Create a PyTorch-enabled ZScoreSustainData object."""
    return TorchZScoreSustainData(data, numStages, backend)


def create_torch_mixture_data(L_yes: np.ndarray, L_no: np.ndarray, numStages: int,
                            backend: Optional[TorchSustainBackend] = None) -> TorchMixtureSustainData:
    """Create a PyTorch-enabled MixtureSustainData object."""
    return TorchMixtureSustainData(L_yes, L_no, numStages, backend)


def create_torch_ordinal_data(prob_nl: np.ndarray, prob_score: np.ndarray, numStages: int,
                            backend: Optional[TorchSustainBackend] = None) -> TorchOrdinalSustainData:
    """Create a PyTorch-enabled OrdinalSustainData object."""
    return TorchOrdinalSustainData(prob_nl, prob_score, numStages, backend)
