###
# GPU-accelerated likelihood computations for fastSuStaIn
# 
# This module provides PyTorch-based implementations of the core likelihood
# computations used in SuStaIn algorithms, with significant performance
# improvements over the original numpy implementations.
#
# Authors: GPU Migration Team
###

import torch
import numpy as np
from typing import Tuple, Optional, Union
from .torch_backend import TorchSustainBackend, TorchMissingDataHandler, safe_torch_operations
from .torch_data_classes import TorchAbstractSustainData


class TorchLikelihoodCalculator:
    """GPU-accelerated likelihood calculator for SuStaIn algorithms."""
    
    def __init__(self, backend: TorchSustainBackend):
        """
        Initialize the likelihood calculator.
        
        Args:
            backend: PyTorch backend for GPU operations
        """
        self.backend = backend
        self.device = backend.device_manager.device
        self.dtype = backend.device_manager.torch_dtype
        self.missing_data_handler = TorchMissingDataHandler(backend)
        
    def calculate_likelihood(self, sustainData: TorchAbstractSustainData, 
                           S: np.ndarray, f: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        GPU-accelerated version of AbstractSustain._calculate_likelihood().
        
        Computes the likelihood of a mixture of models using PyTorch tensors.
        
        Args:
            sustainData: SuStaIn data object (PyTorch-enabled)
            S: Sequence matrix (N_S, N) where N_S=subtypes, N=stages
            f: Fraction vector (N_S,) for each subtype
            
        Returns:
            Tuple of (loglike, total_prob_subj, total_prob_stage, total_prob_cluster, p_perm_k)
        """
        with self.backend.benchmark_operation('calculate_likelihood'):
            # Convert inputs to PyTorch tensors
            S_torch = self.backend.to_torch(S)
            # Ensure f is a numpy array before converting to PyTorch tensor
            f_array = np.asarray(f) if not isinstance(f, np.ndarray) else f
            f_torch = self.backend.to_torch(f_array)
            
            M = sustainData.getNumSamples()
            N_S = S.shape[0]
            N = sustainData.getNumStages()
            
            # Reshape f for broadcasting: (N_S, 1, 1)
            f_reshaped = f_torch.reshape(N_S, 1, 1)
            
            # Create f_val_mat using efficient broadcasting instead of tiling
            # Original: f_val_mat = np.tile(f, (1, N + 1, M))
            # GPU version: Use broadcasting for better memory efficiency
            f_val_mat = f_reshaped.expand(N_S, N + 1, M)  # (N_S, N+1, M)
            # Use permute to achieve the same result as np.transpose(f_val_mat, (2, 1, 0))
            f_val_mat = f_val_mat.permute(2, 1, 0)  # (M, N+1, N_S)
            
            # Initialize p_perm_k tensor
            p_perm_k = torch.zeros((M, N + 1, N_S), device=self.device, dtype=self.dtype)
            
            # Compute likelihood for each subtype
            for s in range(N_S):
                p_perm_k[:, :, s] = self._calculate_likelihood_stage_torch(sustainData, S_torch[s])
            
            # Compute mixture model probabilities using vectorized operations
            # Original: total_prob_cluster = np.squeeze(np.sum(p_perm_k * f_val_mat, 1))
            total_prob_cluster = torch.sum(p_perm_k * f_val_mat, dim=1).squeeze()
            
            # Original: total_prob_stage = np.sum(p_perm_k * f_val_mat, 2)
            total_prob_stage = torch.sum(p_perm_k * f_val_mat, dim=2)
            
            # Original: total_prob_subj = np.sum(total_prob_stage, 1)
            total_prob_subj = torch.sum(total_prob_stage, dim=1)
            
            # Original: loglike = np.sum(np.log(total_prob_subj + 1e-250))
            loglike = torch.sum(safe_torch_operations(total_prob_subj, 'log'))
            
            # Convert results back to numpy for compatibility
            loglike_np = loglike.item()
            total_prob_subj_np = self.backend.to_numpy(total_prob_subj)
            total_prob_stage_np = self.backend.to_numpy(total_prob_stage)
            total_prob_cluster_np = self.backend.to_numpy(total_prob_cluster)
            p_perm_k_np = self.backend.to_numpy(p_perm_k)
            
            return loglike_np, total_prob_subj_np, total_prob_stage_np, total_prob_cluster_np, p_perm_k_np
    
    def _calculate_likelihood_stage_torch(self, sustainData: TorchAbstractSustainData, 
                                        S_single: torch.Tensor) -> torch.Tensor:
        """
        Abstract method for stage-specific likelihood computation.
        Must be implemented by subclasses for specific SuStaIn variants.
        
        Args:
            sustainData: SuStaIn data object
            S_single: Single sequence (N,) for one subtype
            
        Returns:
            Likelihood tensor (M, N+1)
        """
        raise NotImplementedError("Subclasses must implement _calculate_likelihood_stage_torch")


class TorchZScoreLikelihoodCalculator(TorchLikelihoodCalculator):
    """GPU-accelerated likelihood calculator for ZScoreSustain variants."""
    
    def __init__(self, backend: TorchSustainBackend, 
                 stage_biomarker_index: np.ndarray,
                 stage_zscore: np.ndarray,
                 min_biomarker_zscore: np.ndarray,
                 max_biomarker_zscore: np.ndarray,
                 std_biomarker_zscore: np.ndarray):
        """
        Initialize ZScore likelihood calculator.
        
        Args:
            backend: PyTorch backend
            stage_biomarker_index: Biomarker indices for each stage
            stage_zscore: Z-score values for each stage
            min_biomarker_zscore: Minimum z-scores for each biomarker
            max_biomarker_zscore: Maximum z-scores for each biomarker
            std_biomarker_zscore: Standard deviations for each biomarker
        """
        super().__init__(backend)
        
        # Convert model parameters to PyTorch tensors
        self.stage_biomarker_index = self.backend.to_torch(stage_biomarker_index)
        self.stage_zscore = self.backend.to_torch(stage_zscore)
        self.min_biomarker_zscore = self.backend.to_torch(min_biomarker_zscore)
        self.max_biomarker_zscore = self.backend.to_torch(max_biomarker_zscore)
        self.std_biomarker_zscore = self.backend.to_torch(std_biomarker_zscore)
        
        # Ensure stage_biomarker_index has the correct shape (1, N) if it's 1D
        if len(self.stage_biomarker_index.shape) == 1:
            self.stage_biomarker_index = self.stage_biomarker_index.unsqueeze(0)
        
    def _calculate_likelihood_stage_torch(self, sustainData: TorchAbstractSustainData, 
                                        S_single: torch.Tensor) -> torch.Tensor:
        """
        GPU-accelerated version of ZScoreSustain._calculate_likelihood_stage().
        
        Args:
            sustainData: ZScore data object
            S_single: Single sequence (N,) for one subtype
            
        Returns:
            Likelihood tensor (M, N+1)
        """
        with self.backend.benchmark_operation('zscore_likelihood_stage'):
            # stage_biomarker_index is now guaranteed to be 2D with shape (1, N)
            N = self.stage_biomarker_index.shape[1]
            M = sustainData.getNumSamples()
            
            # Convert sequence to inverse mapping
            S_inv = torch.zeros(N, device=self.device, dtype=torch.long)
            S_inv[S_single.long()] = torch.arange(N, device=self.device)
            
            # Get unique biomarkers
            possible_biomarkers = torch.unique(self.stage_biomarker_index)
            B = len(possible_biomarkers)
            
            # Initialize point values
            point_value = torch.zeros((B, N + 2), device=self.device, dtype=self.dtype)
            arange_N = torch.arange(N + 2, device=self.device)
            
            # Compute point values for each biomarker
            for i in range(B):
                b = possible_biomarkers[i]
                
                # Find event locations for this biomarker
                biomarker_mask = (self.stage_biomarker_index == b)
                # Flatten the mask to match S_inv dimensions
                biomarker_mask_flat = biomarker_mask.flatten()
                event_indices = S_inv[biomarker_mask_flat]
                event_location = torch.cat([
                    torch.tensor([0], device=self.device),
                    event_indices,
                    torch.tensor([N], device=self.device)
                ])
                
                # Get event values
                event_values = torch.cat([
                    self.min_biomarker_zscore[i:i+1],
                    self.stage_zscore[biomarker_mask],
                    self.max_biomarker_zscore[i:i+1]
                ])
                
                # Compute point values using vectorized operations
                for j in range(len(event_location) - 1):
                    start_idx = event_location[j]
                    end_idx = event_location[j + 1]
                    
                    if j == 0:
                        # Include start point
                        temp_indices = arange_N[start_idx:end_idx + 2]
                        N_j = end_idx - start_idx + 2
                    else:
                        # Exclude start point
                        temp_indices = arange_N[start_idx + 1:end_idx + 2]
                        N_j = end_idx - start_idx + 1
                    
                    if N_j > 0:
                        # Vectorized linear interpolation
                        start_val = event_values[j]
                        end_val = event_values[j + 1]
                        point_value[i, temp_indices] = torch.linspace(
                            start_val, end_val, N_j, device=self.device, dtype=self.dtype
                        )
            
            # Compute stage values
            stage_value = 0.5 * point_value[:, :-1] + 0.5 * point_value[:, 1:]
            
            # Get data tensor
            data_tensor = sustainData.get_data_torch()
            
            # Compute likelihood using vectorized operations
            # Original: x = (data[:, :, None] - stage_value) / sigmat[None, :, None]
            data_expanded = data_tensor.unsqueeze(2)  # (M, B, 1)
            stage_expanded = stage_value.unsqueeze(0)  # (1, B, N+1)
            sigmat_expanded = self.std_biomarker_zscore.unsqueeze(0).unsqueeze(2)  # (1, B, 1)
            
            x = (data_expanded - stage_expanded) / sigmat_expanded  # (M, B, N+1)
            
            # Compute log-likelihood components
            sqrt_2pi = torch.sqrt(torch.tensor(2.0 * 3.141592653589793, device=self.device, dtype=self.dtype))
            factor = torch.log(1.0 / sqrt_2pi * self.std_biomarker_zscore)
            factor_expanded = factor.unsqueeze(0).unsqueeze(2)  # (1, B, 1)
            coeff = torch.log(torch.tensor(1.0 / float(N + 1), device=self.device, dtype=self.dtype))
            
            # Compute final likelihood
            p_perm_k = coeff + torch.sum(factor_expanded - 0.5 * torch.square(x), dim=1)  # (M, N+1)
            p_perm_k = safe_torch_operations(p_perm_k, 'exp')
            
            return p_perm_k


class TorchZScoreMissingDataLikelihoodCalculator(TorchZScoreLikelihoodCalculator):
    """GPU-accelerated likelihood calculator for ZScoreSustainMissingData."""
    
    def _calculate_likelihood_stage_torch(self, sustainData: TorchAbstractSustainData, 
                                        S_single: torch.Tensor) -> torch.Tensor:
        """
        GPU-accelerated version of ZScoreSustainMissingData._calculate_likelihood_stage().
        
        Handles missing data efficiently using PyTorch operations.
        
        Args:
            sustainData: ZScore data object with potential missing values
            S_single: Single sequence (N,) for one subtype
            
        Returns:
            Likelihood tensor (M, N+1)
        """
        with self.backend.benchmark_operation('zscore_missing_data_likelihood_stage'):
            # stage_biomarker_index is now guaranteed to be 2D with shape (1, N)
            N = self.stage_biomarker_index.shape[1]
            M = sustainData.getNumSamples()
            B = len(torch.unique(self.stage_biomarker_index))
            
            # Compute stage values (same as regular ZScore)
            stage_value = self._compute_stage_values(S_single, N, B)
            
            # Get data tensor
            data_tensor = sustainData.get_data_torch()
            
            # Create missing data probability tensor
            p_missingdata = torch.ones((M, B), device=self.device, dtype=self.dtype)
            p_missingdata = p_missingdata / (self.max_biomarker_zscore - self.min_biomarker_zscore)
            
            # Compute sigmat tensor
            sigmat = self.std_biomarker_zscore.unsqueeze(0).expand(M, -1)  # (M, B)
            
            # Compute factor and coefficient
            sqrt_2pi = torch.sqrt(torch.tensor(2.0 * 3.141592653589793, device=self.device, dtype=self.dtype))
            factor = torch.log(1.0 / sqrt_2pi * self.std_biomarker_zscore)
            factor_expanded = factor.unsqueeze(0).expand(M, -1)  # (M, B)
            coeff = torch.log(torch.tensor(1.0 / float(N + 1), device=self.device, dtype=self.dtype))
            
            # Initialize result tensor
            p_perm_k = torch.zeros((M, N + 1), device=self.device, dtype=self.dtype)
            
            # Compute likelihood for each stage
            for j in range(N + 1):
                # Broadcast stage values
                stage_value_j = stage_value[:, j].unsqueeze(0).expand(M, -1)  # (M, B)
                
                # Compute likelihood for observed data
                x_hasdata = (data_tensor - stage_value_j) / sigmat
                
                # Handle missing data using the missing data handler
                p = self.missing_data_handler.handle_missing_data_likelihood(
                    data_tensor, stage_value_j, sigmat, p_missingdata
                )
                
                # Compute final likelihood for this stage
                p_perm_k[:, j] = coeff + torch.sum(factor_expanded - 0.5 * torch.square(p), dim=1)
            
            # Convert to probabilities
            p_perm_k = safe_torch_operations(p_perm_k, 'exp')
            
            return p_perm_k
    
    def _compute_stage_values(self, S_single: torch.Tensor, N: int, B: int) -> torch.Tensor:
        """Compute stage values for a given sequence."""
        # Convert sequence to inverse mapping
        S_inv = torch.zeros(N, device=self.device, dtype=torch.long)
        S_inv[S_single.long()] = torch.arange(N, device=self.device)
        
        # Get unique biomarkers
        possible_biomarkers = torch.unique(self.stage_biomarker_index)
        
        # Initialize point values
        point_value = torch.zeros((B, N + 2), device=self.device, dtype=self.dtype)
        arange_N = torch.arange(N + 2, device=self.device)
        
        # Compute point values for each biomarker
        for i in range(B):
            b = possible_biomarkers[i]
            
            # Find event locations for this biomarker
            biomarker_mask = (self.stage_biomarker_index == b)
            # Flatten the mask to match S_inv dimensions
            biomarker_mask_flat = biomarker_mask.flatten()
            event_indices = S_inv[biomarker_mask_flat]
            event_location = torch.cat([
                torch.tensor([0], device=self.device),
                event_indices,
                torch.tensor([N], device=self.device)
            ])
            
            # Get event values
            event_values = torch.cat([
                self.min_biomarker_zscore[i:i+1],
                self.stage_zscore[biomarker_mask],
                self.max_biomarker_zscore[i:i+1]
            ])
            
            # Compute point values
            for j in range(len(event_location) - 1):
                start_idx = event_location[j]
                end_idx = event_location[j + 1]
                
                if j == 0:
                    temp_indices = arange_N[start_idx:end_idx + 2]
                    N_j = end_idx - start_idx + 2
                else:
                    temp_indices = arange_N[start_idx + 1:end_idx + 2]
                    N_j = end_idx - start_idx + 1
                
                if N_j > 0:
                    start_val = event_values[j]
                    end_val = event_values[j + 1]
                    point_value[i, temp_indices] = torch.linspace(
                        start_val, end_val, N_j, device=self.device, dtype=self.dtype
                    )
        
        # Compute stage values
        stage_value = 0.5 * point_value[:, :-1] + 0.5 * point_value[:, 1:]
        return stage_value


# Factory functions for creating likelihood calculators
def create_zscore_likelihood_calculator(backend: TorchSustainBackend,
                                      stage_biomarker_index: np.ndarray,
                                      stage_zscore: np.ndarray,
                                      min_biomarker_zscore: np.ndarray,
                                      max_biomarker_zscore: np.ndarray,
                                      std_biomarker_zscore: np.ndarray) -> TorchZScoreLikelihoodCalculator:
    """Create a ZScore likelihood calculator."""
    return TorchZScoreLikelihoodCalculator(
        backend, stage_biomarker_index, stage_zscore,
        min_biomarker_zscore, max_biomarker_zscore, std_biomarker_zscore
    )


def create_zscore_missing_data_likelihood_calculator(backend: TorchSustainBackend,
                                                   stage_biomarker_index: np.ndarray,
                                                   stage_zscore: np.ndarray,
                                                   min_biomarker_zscore: np.ndarray,
                                                   max_biomarker_zscore: np.ndarray,
                                                   std_biomarker_zscore: np.ndarray) -> TorchZScoreMissingDataLikelihoodCalculator:
    """Create a ZScore missing data likelihood calculator."""
    return TorchZScoreMissingDataLikelihoodCalculator(
        backend, stage_biomarker_index, stage_zscore,
        min_biomarker_zscore, max_biomarker_zscore, std_biomarker_zscore
    )


class TorchOrdinalLikelihoodCalculator(TorchLikelihoodCalculator):
    """GPU-accelerated likelihood calculator for OrdinalSustain.

    Vectorization strategy:
        The ordinal likelihood has a sequential dependency — which biomarkers are
        'abnormal' at stage j depends on stages 0..j-1. But this dependency is
        purely index logic that doesn't touch the data. So we:

        1. Precompute masks on CPU: walk through the sequence S to determine,
           for each of the N+1 stages, which prob_score columns (abnormal) and
           which prob_nl columns (normal) to include in the product.

        2. Build a single (M, N+1) log-likelihood tensor on GPU using
           cumulative log-sums — one large kernel instead of N small ones.

        This gives the GPU enough work per call to overcome the kernel launch
        overhead that made the per-stage loop ~11x slower than CPU.
    """

    def __init__(self, backend: TorchSustainBackend,
                 stage_biomarker_index: np.ndarray,
                 stage_score: np.ndarray):
        """
        Initialize Ordinal likelihood calculator.

        Args:
            backend: PyTorch backend
            stage_biomarker_index: Biomarker indices for each stage (1, N)
            stage_score: Score values for each stage (1, N)
        """
        super().__init__(backend)

        # Store numpy versions for CPU-side precomputation
        self.stage_biomarker_index_np = np.asarray(stage_biomarker_index).flatten().astype(int)

        # Convert model parameters to PyTorch tensors
        self.stage_biomarker_index = self.backend.to_torch(stage_biomarker_index)
        self.stage_score = self.backend.to_torch(stage_score)

        # Ensure correct shapes
        if len(self.stage_biomarker_index.shape) == 1:
            self.stage_biomarker_index = self.stage_biomarker_index.unsqueeze(0)
        if len(self.stage_score.shape) == 1:
            self.stage_score = self.stage_score.unsqueeze(0)

    def _calculate_likelihood_stage_torch(self, sustainData: TorchAbstractSustainData,
                                          S_single: torch.Tensor) -> torch.Tensor:
        """
        GPU-accelerated version of OrdinalSustain._calculate_likelihood_stage().

        Precomputes all stage masks on CPU, then does a single batched GPU
        computation for all N+1 stages at once using cumulative log-sums.

        Args:
            sustainData: Ordinal data object
            S_single: Single sequence (N,) for one subtype

        Returns:
            Likelihood tensor (M, N+1) where M=subjects, N=stages
        """
        with self.backend.benchmark_operation('ordinal_likelihood_stage'):
            N = self.stage_score.shape[1]
            M = sustainData.getNumSamples()
            B = sustainData.getNumBiomarkers()

            # Get data tensors (cached on GPU after first access)
            prob_nl_tensor = sustainData.get_prob_nl_torch()    # (M, B)
            prob_score_tensor = sustainData.get_prob_score_torch()  # (M, N_events)

            # ---- Step 1: Precompute masks on CPU (pure index logic, fast) ----
            S_np = S_single.cpu().numpy().astype(int) if S_single.is_cuda else S_single.numpy().astype(int)
            stage_bio_idx = self.stage_biomarker_index_np

            IS_normal = np.ones(B, dtype=bool)
            index_reached = np.zeros(B, dtype=int)

            # For each stage j+1, store which prob_score event indices are abnormal
            # and which prob_nl biomarker indices are normal
            abnormal_event_indices = []  # list of N arrays of event indices
            normal_bio_masks = []        # list of N boolean masks over B biomarkers

            for j in range(N):
                event_idx = S_np[j]
                bio = stage_bio_idx[event_idx]
                index_reached[bio] = event_idx
                IS_normal[bio] = False

                # Snapshot current state
                abnormal_event_indices.append(index_reached[~IS_normal].copy())
                normal_bio_masks.append(IS_normal.copy())

            # ---- Step 2: Compute all log-probabilities on GPU in bulk ----
            # Compute log of all prob columns once (one large GPU kernel)
            log_prob_nl = torch.log(prob_nl_tensor + 1e-250)      # (M, B)
            log_prob_score = torch.log(prob_score_tensor + 1e-250) # (M, N_events)

            coeff = 1.0 / (N + 1)

            # Stage 0: all biomarkers normal → product of all prob_nl columns
            # = sum of all log_prob_nl columns
            log_stage_0 = torch.sum(log_prob_nl, dim=1)  # (M,)

            # Build result tensor
            p_perm_k = torch.zeros((M, N + 1), device=self.device, dtype=self.dtype)
            p_perm_k[:, 0] = coeff * torch.exp(log_stage_0)

            # Stages 1..N: use precomputed masks to gather and sum logs
            for j in range(N):
                abn_idx = abnormal_event_indices[j]
                nrm_mask = normal_bio_masks[j]

                # Sum log-probs for abnormal events
                if len(abn_idx) > 0:
                    abn_idx_t = torch.tensor(abn_idx, device=self.device, dtype=torch.long)
                    log_abn = torch.sum(log_prob_score[:, abn_idx_t], dim=1)  # (M,)
                else:
                    log_abn = torch.zeros(M, device=self.device, dtype=self.dtype)

                # Sum log-probs for normal biomarkers
                if nrm_mask.any():
                    nrm_idx_t = torch.tensor(np.where(nrm_mask)[0], device=self.device, dtype=torch.long)
                    log_nrm = torch.sum(log_prob_nl[:, nrm_idx_t], dim=1)  # (M,)
                else:
                    log_nrm = torch.zeros(M, device=self.device, dtype=self.dtype)

                p_perm_k[:, j + 1] = coeff * torch.exp(log_abn + log_nrm)

            return p_perm_k


def create_ordinal_likelihood_calculator(backend: TorchSustainBackend,
                                        stage_biomarker_index: np.ndarray,
                                        stage_score: np.ndarray) -> TorchOrdinalLikelihoodCalculator:
    """Create an Ordinal likelihood calculator."""
    return TorchOrdinalLikelihoodCalculator(backend, stage_biomarker_index, stage_score)
