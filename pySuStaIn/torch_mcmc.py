###
# GPU-batched MCMC for OrdinalSustain
#
# This module implements the MCMC loop entirely on GPU, eliminating
# per-iteration CPU<->GPU transfer overhead.
###

import torch
import numpy as np
from typing import Tuple, Optional
from tqdm.auto import tqdm


class TorchOrdinalMCMC:
    """GPU-batched MCMC for OrdinalSustain."""
    
    def __init__(self, 
                 torch_sustain_data,
                 stage_biomarker_index: torch.Tensor,
                 stage_score: torch.Tensor,
                 device: torch.device,
                 dtype: torch.dtype = torch.float32):
        """
        Args:
            torch_sustain_data: TorchOrdinalSustainData object with prob_nl, prob_score on GPU
            stage_biomarker_index: (1, N) tensor of biomarker indices per stage
            stage_score: (1, N) tensor of score values per stage
            device: torch device (cuda or cpu)
            dtype: torch dtype
        """
        self.data = torch_sustain_data
        self.device = device
        self.dtype = dtype
        
        # Ensure tensors are on correct device
        if isinstance(stage_biomarker_index, np.ndarray):
            stage_biomarker_index = torch.from_numpy(stage_biomarker_index)
        if isinstance(stage_score, np.ndarray):
            stage_score = torch.from_numpy(stage_score)
            
        self.stage_biomarker_index = stage_biomarker_index.to(device)
        self.stage_score = stage_score.float().to(device)
        
        self.N = self.stage_score.shape[1]  # number of stages
        self.M = torch_sustain_data.getNumSamples()  # number of subjects
        self.B = torch_sustain_data.getNumBiomarkers()  # number of biomarkers
        
        # Pre-fetch data tensors to avoid repeated calls
        self._prob_nl = torch_sustain_data.get_prob_nl_torch()  # (M, B)
        self._prob_score = torch_sustain_data.get_prob_score_torch()  # (M, N)
        
    def _calculate_likelihood_stage_batch(self, S: torch.Tensor) -> torch.Tensor:
        """
        Calculate likelihood for a single sequence - fully on GPU.
        
        Args:
            S: (N,) tensor - single sequence
            
        Returns:
            p_perm_k: (M, N+1) tensor - likelihood per subject per stage
        """
        N = self.N
        M = self.M
        B = self.B
        
        prob_nl = self._prob_nl  # (M, B)
        prob_score = self._prob_score  # (M, N)
        
        # Initialize tracking tensors
        IS_normal = torch.ones(B, device=self.device, dtype=self.dtype)
        IS_abnormal = torch.zeros(B, device=self.device, dtype=self.dtype)
        index_reached = torch.zeros(B, device=self.device, dtype=torch.long)
        
        # Initialize result
        p_perm_k = torch.zeros((M, N + 1), device=self.device, dtype=self.dtype)
        
        # Stage 0: all biomarkers normal
        coeff = 1.0 / (N + 1)
        p_perm_k[:, 0] = coeff * torch.prod(prob_nl, dim=1)
        
        # Loop over stages
        for j in range(N):
            index_justreached = int(S[j].item())
            biomarker_justreached = int(self.stage_biomarker_index[0, index_justreached].item())
            
            index_reached[biomarker_justreached] = index_justreached
            IS_normal[biomarker_justreached] = 0
            IS_abnormal[biomarker_justreached] = 1
            
            bool_IS_normal = IS_normal.bool()
            bool_IS_abnormal = IS_abnormal.bool()
            
            # Get indices for abnormal biomarkers
            abnormal_indices = index_reached[bool_IS_abnormal]
            
            # Compute probabilities
            if abnormal_indices.numel() > 0:
                prob_abnormal = prob_score[:, abnormal_indices]
                prod_prob_abnormal = torch.prod(prob_abnormal, dim=1)
            else:
                prod_prob_abnormal = torch.ones(M, device=self.device, dtype=self.dtype)
            
            if bool_IS_normal.any():
                prob_normal = prob_nl[:, bool_IS_normal]
                prod_prob_normal = torch.prod(prob_normal, dim=1)
            else:
                prod_prob_normal = torch.ones(M, device=self.device, dtype=self.dtype)
            
            p_perm_k[:, j + 1] = coeff * prod_prob_abnormal * prod_prob_normal
        
        return p_perm_k
    
    def _calculate_likelihood_batch(self,
                                    S: torch.Tensor,
                                    f: torch.Tensor) -> torch.Tensor:
        """
        Calculate full likelihood for mixture model - fully on GPU.
        
        Args:
            S: (N_S, N) tensor - sequences for each subtype
            f: (N_S,) tensor - fractions for each subtype
            
        Returns:
            loglike: scalar tensor
        """
        N_S = S.shape[0]
        N = self.N
        M = self.M
        
        # Reshape f for broadcasting
        f_reshaped = f.reshape(N_S, 1, 1)
        f_val_mat = f_reshaped.expand(N_S, N + 1, M).permute(2, 1, 0)  # (M, N+1, N_S)
        
        # Compute p_perm_k for each subtype
        p_perm_k = torch.zeros((M, N + 1, N_S), device=self.device, dtype=self.dtype)
        for s in range(N_S):
            p_perm_k[:, :, s] = self._calculate_likelihood_stage_batch(S[s])
        
        # Compute mixture probabilities
        total_prob_stage = torch.sum(p_perm_k * f_val_mat, dim=2)
        total_prob_subj = torch.sum(total_prob_stage, dim=1)
        loglike = torch.sum(torch.log(total_prob_subj + 1e-250))
        
        return loglike
    
    def _perturb_sequence_gpu(self, 
                              current_seq: torch.Tensor,
                              seq_sigma: torch.Tensor,
                              generator: torch.Generator) -> torch.Tensor:
        """
        Perturb a sequence while respecting biomarker ordering constraints.
        All operations on GPU.
        """
        N = len(current_seq)
        
        # Select random event to move
        move_event_from = torch.randint(0, N, (1,), device=self.device, generator=generator).item()
        
        # Get current location mapping
        current_location = torch.zeros(N, device=self.device, dtype=torch.long)
        current_location[current_seq.long()] = torch.arange(N, device=self.device)
        
        selected_event = int(current_seq[move_event_from].item())
        this_stage_score = self.stage_score[0, selected_event]
        selected_biomarker = self.stage_biomarker_index[0, selected_event]
        
        # Find valid move range based on biomarker constraints
        biomarker_mask = (self.stage_biomarker_index[0] == selected_biomarker)
        possible_scores = self.stage_score[0, biomarker_mask]
        
        # Find bounds
        min_filter = possible_scores < this_stage_score
        max_filter = possible_scores > this_stage_score
        
        events = torch.arange(N, device=self.device)
        
        if min_filter.any():
            min_score_bound = possible_scores[min_filter].max()
            min_bound_mask = (self.stage_score[0] == min_score_bound) & biomarker_mask
            if min_bound_mask.any():
                min_bound_event = events[min_bound_mask][0]
                move_to_lower = current_location[min_bound_event].item() + 1
            else:
                move_to_lower = 0
        else:
            move_to_lower = 0
        
        if max_filter.any():
            max_score_bound = possible_scores[max_filter].min()
            max_bound_mask = (self.stage_score[0] == max_score_bound) & biomarker_mask
            if max_bound_mask.any():
                max_bound_event = events[max_bound_mask][0]
                move_to_upper = current_location[max_bound_event].item()
            else:
                move_to_upper = N
        else:
            move_to_upper = N
        
        # Sample new position
        if move_to_lower >= move_to_upper:
            return current_seq  # Can't move
        
        possible_positions = torch.arange(move_to_lower, move_to_upper, device=self.device)
        distance = possible_positions.float() - move_event_from
        
        this_sigma = seq_sigma[selected_event] if seq_sigma.numel() > 1 else seq_sigma
        if this_sigma < 0.01:
            this_sigma = 0.01
            
        weight = torch.exp(-0.5 * (distance / this_sigma) ** 2)
        weight = weight / weight.sum()
        
        idx = torch.multinomial(weight, 1, generator=generator).item()
        move_event_to = possible_positions[idx].item()
        
        # Apply move by creating new sequence
        seq_list = current_seq.tolist()
        event_to_move = seq_list.pop(move_event_from)
        seq_list.insert(move_event_to, event_to_move)
        new_seq = torch.tensor(seq_list, device=self.device, dtype=current_seq.dtype)
        
        return new_seq
    
    def perform_mcmc(self,
                     seq_init: np.ndarray,
                     f_init: np.ndarray,
                     n_iterations: int,
                     seq_sigma: np.ndarray,
                     f_sigma: np.ndarray,
                     seed: int = None) -> Tuple[np.ndarray, np.ndarray, float,
                                                 np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform MCMC sampling entirely on GPU.
        
        Returns numpy arrays for compatibility with existing code.
        """
        # Set up generator for reproducibility
        generator = torch.Generator(device=self.device)
        if seed is not None:
            generator.manual_seed(seed)
        
        N_S = seq_init.shape[0]
        N = self.N
        
        # Convert inputs to GPU tensors
        current_seq = torch.from_numpy(seq_init.copy()).to(self.device).float()
        current_f = torch.from_numpy(f_init.copy()).to(self.device).float()
        
        if isinstance(seq_sigma, (int, float)):
            seq_sigma_t = torch.full((N_S, N), float(seq_sigma), device=self.device, dtype=self.dtype)
        else:
            seq_sigma_t = torch.from_numpy(seq_sigma.astype(np.float32)).to(self.device)
            
        if isinstance(f_sigma, (int, float)):
            f_sigma_t = torch.full((N_S,), float(f_sigma), device=self.device, dtype=self.dtype)
        else:
            f_sigma_t = torch.from_numpy(f_sigma.astype(np.float32)).to(self.device)
        
        # Allocate output tensors on GPU
        samples_sequence = torch.zeros((N_S, N, n_iterations), device=self.device, dtype=torch.float32)
        samples_f = torch.zeros((N_S, n_iterations), device=self.device, dtype=self.dtype)
        samples_likelihood = torch.zeros(n_iterations, device=self.device, dtype=self.dtype)
        
        # Initialize
        samples_sequence[:, :, 0] = current_seq
        samples_f[:, 0] = current_f
        
        # Calculate initial likelihood
        samples_likelihood[0] = self._calculate_likelihood_batch(current_seq, current_f)
        
        # Reduce tqdm update frequency for large iterations
        tqdm_update_iters = int(n_iterations/1000) if n_iterations > 100000 else None
        
        # MCMC loop - all operations on GPU
        for i in tqdm(range(1, n_iterations), "MCMC Iteration", n_iterations - 1, miniters=tqdm_update_iters):
            # Start from previous state
            new_seq = samples_sequence[:, :, i - 1].clone()
            new_f = samples_f[:, i - 1].clone()
            
            # Perturb sequences for each subtype
            seq_order = torch.randperm(N_S, device=self.device, generator=generator)
            for s_idx in seq_order:
                s = s_idx.item()
                new_seq[s] = self._perturb_sequence_gpu(new_seq[s], seq_sigma_t[s], generator)
            
            # Perturb fractions
            noise = torch.randn(N_S, device=self.device, generator=generator, dtype=self.dtype)
            new_f = new_f + f_sigma_t * noise
            new_f = torch.abs(new_f)
            new_f = new_f / torch.sum(new_f)
            
            # Calculate new likelihood
            new_likelihood = self._calculate_likelihood_batch(new_seq, new_f)
            
            # Accept/reject
            ratio = torch.exp(new_likelihood - samples_likelihood[i - 1])
            rand_val = torch.rand(1, device=self.device, generator=generator)
            
            if ratio >= rand_val:
                # Accept
                samples_sequence[:, :, i] = new_seq
                samples_f[:, i] = new_f
                samples_likelihood[i] = new_likelihood
            else:
                # Reject - keep previous
                samples_sequence[:, :, i] = samples_sequence[:, :, i - 1]
                samples_f[:, i] = samples_f[:, i - 1]
                samples_likelihood[i] = samples_likelihood[i - 1]
        
        # Find ML solution
        ml_idx = torch.argmax(samples_likelihood).item()
        ml_likelihood = samples_likelihood[ml_idx].item()
        ml_sequence = samples_sequence[:, :, ml_idx].cpu().numpy()
        ml_f = samples_f[:, ml_idx].cpu().numpy()
        
        # Convert all results to numpy
        samples_sequence_np = samples_sequence.cpu().numpy()
        samples_f_np = samples_f.cpu().numpy()
        samples_likelihood_np = samples_likelihood.cpu().numpy().reshape(-1, 1)
        
        return ml_sequence, ml_f, ml_likelihood, samples_sequence_np, samples_f_np, samples_likelihood_np
