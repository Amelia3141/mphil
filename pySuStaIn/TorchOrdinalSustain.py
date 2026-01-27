###
# GPU-accelerated OrdinalSustain implementation
#
# This module provides a PyTorch-based implementation of OrdinalSustain
# with significant performance improvements through GPU acceleration.
#
# Key optimization: The _perform_mcmc method keeps all computation on GPU
# throughout the MCMC loop, only converting to numpy at the very end.
# This avoids the per-iteration CPU<->GPU transfer overhead that made
# the naive GPU implementation slower than CPU.
#
# Authors: GPU Migration Team
###

import torch
import numpy as np
from tqdm.auto import tqdm
from typing import Optional, Tuple, Union
from .OrdinalSustain import OrdinalSustain, OrdinalSustainData
from .torch_backend import TorchSustainBackend, create_torch_backend
from .torch_data_classes import TorchOrdinalSustainData, create_torch_ordinal_data
from .torch_likelihood import TorchOrdinalLikelihoodCalculator, create_ordinal_likelihood_calculator


class TorchOrdinalSustain(OrdinalSustain):
    """
    GPU-accelerated version of OrdinalSustain.

    This class extends the original OrdinalSustain with PyTorch-based
    GPU acceleration while maintaining full compatibility with the original API.
    
    The key optimization is in _perform_mcmc which keeps all data on GPU
    throughout the MCMC loop, avoiding per-iteration transfer overhead.
    """

    def __init__(self,
                 prob_nl: np.ndarray,
                 prob_score: np.ndarray,
                 score_vals: np.ndarray,
                 biomarker_labels: list,
                 N_startpoints: int,
                 N_S_max: int,
                 N_iterations_MCMC: int,
                 output_folder: str,
                 dataset_name: str,
                 use_parallel_startpoints: bool,
                 seed: Optional[int] = None,
                 use_gpu: bool = True,
                 device_id: Optional[int] = None):
        """
        Initialize GPU-accelerated OrdinalSustain.
        """
        # Initialize the original class first
        super().__init__(
            prob_nl, prob_score, score_vals, biomarker_labels,
            N_startpoints, N_S_max, N_iterations_MCMC,
            output_folder, dataset_name, use_parallel_startpoints, seed
        )

        # Initialize PyTorch backend
        self.torch_backend = create_torch_backend(use_gpu=use_gpu, device_id=device_id)
        self.use_gpu = self.torch_backend.use_gpu
        self.device = self.torch_backend.device_manager.device
        self.dtype = self.torch_backend.device_manager.torch_dtype

        # Access the sustainData attribute from parent class
        sustain_data = None
        for attr_name in ['_OrdinalSustain__sustainData', '__sustainData', '_sustainData', 'sustainData']:
            sustain_data = getattr(self, attr_name, None)
            if sustain_data is not None:
                break

        if sustain_data is None:
            raise AttributeError("Could not find sustainData attribute.")

        # Store reference for GPU methods
        self._sustain_data_ref = sustain_data

        # Create PyTorch-enabled data object
        # IMPORTANT: Use the RESHAPED prob_score from sustainData, not the original input!
        self.torch_sustain_data = create_torch_ordinal_data(
            sustain_data.prob_nl, sustain_data.prob_score, sustain_data.getNumStages(), self.torch_backend
        )

        # Create GPU-accelerated likelihood calculator
        self.torch_likelihood_calculator = create_ordinal_likelihood_calculator(
            self.torch_backend,
            self.stage_biomarker_index,
            self.stage_score
        )

        # Pre-convert stage indices to torch for MCMC
        self.stage_score_torch = torch.tensor(self.stage_score, device=self.device, dtype=self.dtype)
        self.stage_biomarker_index_torch = torch.tensor(self.stage_biomarker_index, device=self.device, dtype=torch.long)

        print(f"TorchOrdinalSustain initialized with {'GPU' if self.use_gpu else 'CPU'} acceleration")

    def _calculate_likelihood_torch(self, S: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        """
        GPU-only likelihood computation - returns torch tensor, no numpy conversion.
        
        This is the key method for GPU speedup: it stays entirely on GPU.
        
        Args:
            S: Sequence matrix (N_S, N) as torch tensor
            f: Fraction vector (N_S,) as torch tensor
            
        Returns:
            Log-likelihood as torch scalar tensor
        """
        M = self.torch_sustain_data.getNumSamples()
        N_S = S.shape[0]
        N = self.torch_sustain_data.getNumStages()
        
        # Reshape f for broadcasting: (N_S, 1, 1) -> (M, N+1, N_S)
        f_reshaped = f.reshape(N_S, 1, 1)
        f_val_mat = f_reshaped.expand(N_S, N + 1, M).permute(2, 1, 0)
        
        # Initialize p_perm_k tensor
        p_perm_k = torch.zeros((M, N + 1, N_S), device=self.device, dtype=self.dtype)
        
        # Compute likelihood for each subtype
        for s in range(N_S):
            p_perm_k[:, :, s] = self._calculate_likelihood_stage_torch(S[s])
        
        # Compute mixture model probabilities
        total_prob_stage = torch.sum(p_perm_k * f_val_mat, dim=2)
        total_prob_subj = torch.sum(total_prob_stage, dim=1)
        
        # Return log-likelihood (stays on GPU)
        loglike = torch.sum(torch.log(total_prob_subj + 1e-250))
        
        return loglike

    def _calculate_likelihood_stage_torch(self, S_single: torch.Tensor) -> torch.Tensor:
        """
        GPU-only stage likelihood computation.
        
        Args:
            S_single: Single sequence (N,) as torch tensor
            
        Returns:
            Likelihood tensor (M, N+1) on GPU
        """
        N = self.stage_score_torch.shape[1]
        B = self.torch_sustain_data.getNumBiomarkers()
        M = self.torch_sustain_data.getNumSamples()
        
        prob_nl_tensor = self.torch_sustain_data.get_prob_nl_torch()
        prob_score_tensor = self.torch_sustain_data.get_prob_score_torch()
        
        # Initialize state tracking
        IS_normal = torch.ones(B, device=self.device, dtype=self.dtype)
        IS_abnormal = torch.zeros(B, device=self.device, dtype=self.dtype)
        index_reached = torch.zeros(B, device=self.device, dtype=torch.long)
        
        # Initialize result
        p_perm_k = torch.zeros((M, N + 1), device=self.device, dtype=self.dtype)
        coeff = 1.0 / (N + 1)
        
        # Stage 0: all biomarkers normal
        p_perm_k[:, 0] = coeff * torch.prod(prob_nl_tensor, dim=1)
        
        # Loop over stages
        for j in range(N):
            index_justreached = S_single[j].long().item()
            biomarker_justreached = self.stage_biomarker_index_torch[0, index_justreached].item()
            
            index_reached[biomarker_justreached] = index_justreached
            IS_normal[biomarker_justreached] = 0
            IS_abnormal[biomarker_justreached] = 1
            
            bool_IS_normal = IS_normal.bool()
            bool_IS_abnormal = IS_abnormal.bool()
            
            # Get indices for abnormal biomarkers
            abnormal_indices = index_reached[bool_IS_abnormal]
            
            # Compute probabilities
            if abnormal_indices.numel() > 0:
                prob_abnormal = prob_score_tensor[:, abnormal_indices]
                prod_prob_abnormal = torch.prod(prob_abnormal, dim=1)
            else:
                prod_prob_abnormal = torch.ones(M, device=self.device, dtype=self.dtype)
            
            if bool_IS_normal.any():
                prob_normal = prob_nl_tensor[:, bool_IS_normal]
                prod_prob_normal = torch.prod(prob_normal, dim=1)
            else:
                prod_prob_normal = torch.ones(M, device=self.device, dtype=self.dtype)
            
            p_perm_k[:, j + 1] = coeff * prod_prob_abnormal * prod_prob_normal
        
        return p_perm_k

    def _perform_mcmc(self, sustainData, seq_init, f_init, n_iterations, seq_sigma, f_sigma, rng=None):
        """
        GPU-batched MCMC - keeps all computation on GPU until final return.
        
        This overrides the parent _perform_mcmc to avoid per-iteration CPU<->GPU transfers.
        All intermediate computation stays on GPU; only converts to numpy at return.
        """
        if not self.use_gpu:
            # Fall back to CPU implementation
            return super()._perform_mcmc(sustainData, seq_init, f_init, n_iterations, seq_sigma, f_sigma, rng)
        
        # Use provided rng or fall back to global_rng
        if rng is None:
            rng = self.global_rng
        
        N = self.stage_score.shape[1]
        N_S = seq_init.shape[0]
        
        # Set torch random seed for reproducibility
        torch_seed = rng.integers(2**31)
        torch.manual_seed(torch_seed)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed(torch_seed)
        
        # Handle seq_sigma and f_sigma
        if isinstance(f_sigma, float):
            f_sigma = np.array([f_sigma])
        if isinstance(seq_sigma, (int, float)):
            seq_sigma_np = np.full((N_S, N), seq_sigma)
        else:
            seq_sigma_np = seq_sigma
        
        # Convert to torch tensors on GPU
        seq_sigma_torch = torch.tensor(seq_sigma_np, device=self.device, dtype=self.dtype)
        f_sigma_torch = torch.tensor(f_sigma, device=self.device, dtype=self.dtype)
        
        # Initialize sample storage on GPU
        samples_sequence = torch.zeros((N_S, N, n_iterations), device=self.device, dtype=torch.long)
        samples_f = torch.zeros((N_S, n_iterations), device=self.device, dtype=self.dtype)
        samples_likelihood = torch.zeros(n_iterations, device=self.device, dtype=self.dtype)
        
        # Initialize current state on GPU
        current_sequence = torch.tensor(seq_init, device=self.device, dtype=torch.long)
        current_f = torch.tensor(f_init, device=self.device, dtype=self.dtype).flatten()
        
        samples_sequence[:, :, 0] = current_sequence
        samples_f[:, 0] = current_f
        
        # Pre-compute stage constraints for MCMC proposals
        stage_score_flat = self.stage_score_torch.flatten()
        stage_biomarker_flat = self.stage_biomarker_index_torch.flatten()
        
        # Reduce tqdm update frequency for large iterations
        tqdm_update_iters = int(n_iterations/1000) if n_iterations > 100000 else None
        
        for i in tqdm(range(n_iterations), "MCMC Iteration", n_iterations, miniters=tqdm_update_iters):
            if i > 0:
                # Generate random permutation of subtypes
                seq_order = torch.randperm(N_S, device=self.device)
                
                for s_idx in range(N_S):
                    s = seq_order[s_idx].item()
                    
                    # Choose random event to move
                    move_event_from = torch.randint(0, N, (1,), device=self.device).item()
                    
                    # Get current sequence for this subtype
                    current_seq_s = samples_sequence[s, :, i - 1].clone()
                    
                    # Build current location mapping
                    current_location = torch.zeros(N, device=self.device, dtype=torch.long)
                    current_location[current_seq_s] = torch.arange(N, device=self.device)
                    
                    selected_event = current_seq_s[move_event_from].item()
                    this_stage_score = stage_score_flat[selected_event]
                    selected_biomarker = stage_biomarker_flat[selected_event]
                    
                    # Find possible positions respecting monotonicity constraints
                    same_biomarker_mask = (stage_biomarker_flat == selected_biomarker)
                    possible_scores = stage_score_flat[same_biomarker_mask]
                    
                    # Find bounds
                    lower_scores = possible_scores[possible_scores < this_stage_score]
                    upper_scores = possible_scores[possible_scores > this_stage_score]
                    
                    if lower_scores.numel() > 0:
                        min_score_bound = lower_scores.max()
                        # Find event with this score for this biomarker
                        bound_mask = (stage_score_flat == min_score_bound) & same_biomarker_mask
                        bound_event = torch.where(bound_mask)[0]
                        if bound_event.numel() > 0:
                            move_event_to_lower_bound = current_location[bound_event[0]].item() + 1
                        else:
                            move_event_to_lower_bound = 0
                    else:
                        move_event_to_lower_bound = 0
                    
                    if upper_scores.numel() > 0:
                        max_score_bound = upper_scores.min()
                        bound_mask = (stage_score_flat == max_score_bound) & same_biomarker_mask
                        bound_event = torch.where(bound_mask)[0]
                        if bound_event.numel() > 0:
                            move_event_to_upper_bound = current_location[bound_event[0]].item()
                        else:
                            move_event_to_upper_bound = N
                    else:
                        move_event_to_upper_bound = N
                    
                    # Handle edge case
                    if move_event_to_lower_bound >= move_event_to_upper_bound:
                        samples_sequence[s, :, i] = current_seq_s
                        continue
                    
                    # Generate possible positions
                    possible_positions = torch.arange(move_event_to_lower_bound, move_event_to_upper_bound, device=self.device)
                    
                    if possible_positions.numel() == 0:
                        samples_sequence[s, :, i] = current_seq_s
                        continue
                    
                    # Compute weights based on distance
                    distance = possible_positions.float() - move_event_from
                    this_seq_sigma = seq_sigma_torch[s, selected_event]
                    
                    # Normal PDF weights
                    weight = torch.exp(-0.5 * (distance / this_seq_sigma) ** 2)
                    weight = weight / weight.sum()
                    
                    # Sample new position
                    idx = torch.multinomial(weight, 1).item()
                    move_event_to = possible_positions[idx].item()
                    
                    # Create new sequence
                    new_sequence = torch.cat([
                        current_seq_s[:move_event_from],
                        current_seq_s[move_event_from + 1:]
                    ])
                    new_sequence = torch.cat([
                        new_sequence[:move_event_to],
                        torch.tensor([selected_event], device=self.device, dtype=torch.long),
                        new_sequence[move_event_to:]
                    ])
                    
                    samples_sequence[s, :, i] = new_sequence
                
                # Update f with Gaussian perturbation
                new_f = samples_f[:, i - 1] + f_sigma_torch * torch.randn(N_S, device=self.device, dtype=self.dtype)
                new_f = torch.abs(new_f)
                new_f = new_f / new_f.sum()
                samples_f[:, i] = new_f
            
            # Calculate likelihood (stays on GPU!)
            S = samples_sequence[:, :, i].float()
            f = samples_f[:, i]
            likelihood_sample = self._calculate_likelihood_torch(S, f)
            samples_likelihood[i] = likelihood_sample
            
            # Metropolis accept/reject
            if i > 0:
                ratio = torch.exp(samples_likelihood[i] - samples_likelihood[i - 1])
                if ratio < torch.rand(1, device=self.device):
                    # Reject: revert to previous
                    samples_likelihood[i] = samples_likelihood[i - 1]
                    samples_sequence[:, :, i] = samples_sequence[:, :, i - 1]
                    samples_f[:, i] = samples_f[:, i - 1]
        
        # Find ML sample
        ml_idx = torch.argmax(samples_likelihood).item()
        ml_likelihood = samples_likelihood.max()
        ml_sequence = samples_sequence[:, :, ml_idx]
        ml_f = samples_f[:, ml_idx]
        
        # Convert to numpy only at the very end
        return (
            ml_sequence.cpu().numpy().astype(float),
            ml_f.cpu().numpy(),
            ml_likelihood.item(),
            samples_sequence.cpu().numpy().astype(float),
            samples_f.cpu().numpy(),
            samples_likelihood.cpu().numpy().reshape(-1, 1)
        )

    def _calculate_likelihood(self, sustainData, S: np.ndarray, f: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        GPU-accelerated likelihood computation (numpy interface for compatibility).
        """
        if self.use_gpu:
            try:
                return self.torch_likelihood_calculator.calculate_likelihood(
                    self.torch_sustain_data, S, f
                )
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("GPU out of memory, falling back to CPU computation")
                    self.torch_backend.clear_cache()
                    return super()._calculate_likelihood(sustainData, S, f)
                else:
                    raise
        else:
            return super()._calculate_likelihood(sustainData, S, f)

    def _calculate_likelihood_stage(self, sustainData, S: np.ndarray) -> np.ndarray:
        """
        GPU-accelerated stage likelihood computation (numpy interface).
        """
        if self.use_gpu:
            try:
                S_torch = self.torch_backend.to_torch(S)
                result_torch = self.torch_likelihood_calculator._calculate_likelihood_stage_torch(
                    self.torch_sustain_data, S_torch
                )
                return self.torch_backend.to_numpy(result_torch)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("GPU out of memory, falling back to CPU computation")
                    self.torch_backend.clear_cache()
                    return super()._calculate_likelihood_stage(sustainData, S)
                else:
                    raise
        else:
            return super()._calculate_likelihood_stage(sustainData, S)

    def get_performance_stats(self) -> dict:
        """Get performance statistics from GPU computations."""
        return self.torch_backend.get_performance_stats()

    def clear_gpu_cache(self):
        """Clear GPU memory cache."""
        if self.use_gpu:
            self.torch_backend.clear_cache()

    def switch_to_cpu(self):
        """Switch to CPU-only computation."""
        self.use_gpu = False
        print("Switched to CPU-only computation")

    def switch_to_gpu(self, device_id: Optional[int] = None):
        """Switch to GPU computation."""
        try:
            self.torch_backend = create_torch_backend(use_gpu=True, device_id=device_id)
            self.use_gpu = self.torch_backend.use_gpu
            self.device = self.torch_backend.device_manager.device

            if self.use_gpu:
                sustain_data = self._sustain_data_ref

                self.torch_sustain_data = create_torch_ordinal_data(
                    sustain_data.prob_nl, sustain_data.prob_score,
                    sustain_data.getNumStages(), self.torch_backend
                )

                self.torch_likelihood_calculator = create_ordinal_likelihood_calculator(
                    self.torch_backend,
                    self.stage_biomarker_index,
                    self.stage_score
                )
                print("Switched to GPU computation")
            else:
                print("GPU not available, staying on CPU")
        except Exception as e:
            print(f"Failed to switch to GPU: {e}")
            self.use_gpu = False


# Factory function
def create_torch_ordinal_sustain(
    prob_nl: np.ndarray,
    prob_score: np.ndarray,
    score_vals: np.ndarray,
    biomarker_labels: list,
    N_startpoints: int = 25,
    N_S_max: int = 3,
    N_iterations_MCMC: int = 100000,
    output_folder: str = "./output",
    dataset_name: str = "dataset",
    use_parallel_startpoints: bool = True,
    seed: Optional[int] = None,
    use_gpu: bool = True,
    device_id: Optional[int] = None
) -> TorchOrdinalSustain:
    """Factory function to create a GPU-accelerated OrdinalSustain instance."""
    return TorchOrdinalSustain(
        prob_nl, prob_score, score_vals, biomarker_labels,
        N_startpoints, N_S_max, N_iterations_MCMC,
        output_folder, dataset_name, use_parallel_startpoints,
        seed, use_gpu, device_id
    )
