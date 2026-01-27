###
# GPU-accelerated OrdinalSustain using torch.vmap for true parallelisation
#
# Key: Use torch.vmap to vectorise likelihood computation across batch dimension,
# eliminating Python loops and allowing GPU to process all K sequences in parallel.
###

import torch
import numpy as np
from tqdm.auto import tqdm
from typing import Optional
from functools import partial
from .OrdinalSustain import OrdinalSustain
from .torch_backend import create_torch_backend
from .torch_data_classes import create_torch_ordinal_data
from .torch_likelihood import create_ordinal_likelihood_calculator


class TorchOrdinalSustain(OrdinalSustain):
    """GPU-accelerated OrdinalSustain with vmap-batched likelihood."""

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
        
        super().__init__(
            prob_nl, prob_score, score_vals, biomarker_labels,
            N_startpoints, N_S_max, N_iterations_MCMC,
            output_folder, dataset_name, use_parallel_startpoints, seed
        )

        self.torch_backend = create_torch_backend(use_gpu=use_gpu, device_id=device_id)
        self.use_gpu = self.torch_backend.use_gpu
        self.device = self.torch_backend.device_manager.device
        self.dtype = self.torch_backend.device_manager.torch_dtype

        # Get sustainData
        sustain_data = None
        for attr_name in ['_OrdinalSustain__sustainData', '__sustainData', '_sustainData', 'sustainData']:
            sustain_data = getattr(self, attr_name, None)
            if sustain_data is not None:
                break
        if sustain_data is None:
            raise AttributeError("Could not find sustainData")
        self._sustain_data_ref = sustain_data

        self.torch_sustain_data = create_torch_ordinal_data(
            sustain_data.prob_nl, sustain_data.prob_score, 
            sustain_data.getNumStages(), self.torch_backend
        )
        self.torch_likelihood_calculator = create_ordinal_likelihood_calculator(
            self.torch_backend, self.stage_biomarker_index, self.stage_score
        )

        # Pre-load to GPU
        self.prob_nl_gpu = torch.tensor(sustain_data.prob_nl, device=self.device, dtype=self.dtype)
        self.prob_score_gpu = torch.tensor(sustain_data.prob_score, device=self.device, dtype=self.dtype)
        self.stage_biomarker_index_flat = torch.tensor(
            self.stage_biomarker_index.flatten(), device=self.device, dtype=torch.long
        )
        self.N_stages = self.stage_score.shape[1]
        self.M_subjects = sustain_data.prob_nl.shape[0]
        self.B_biomarkers = sustain_data.prob_nl.shape[1]

        print(f"TorchOrdinalSustain: {'GPU' if self.use_gpu else 'CPU'}")

    def _single_sequence_likelihood(self, S_single: torch.Tensor) -> torch.Tensor:
        """
        Compute likelihood for a single sequence. Used with vmap.
        
        Args:
            S_single: (N,) sequence
            
        Returns:
            (M, N+1) likelihood matrix
        """
        N = self.N_stages
        M = self.M_subjects
        B = self.B_biomarkers
        
        p_perm_k = torch.zeros((M, N + 1), device=self.device, dtype=self.dtype)
        coeff = 1.0 / (N + 1)
        
        # Stage 0
        p_perm_k[:, 0] = coeff * torch.prod(self.prob_nl_gpu, dim=1)
        
        # Track state - must be done sequentially
        IS_normal = torch.ones(B, device=self.device, dtype=torch.bool)
        index_reached = torch.zeros(B, device=self.device, dtype=torch.long)
        
        for j in range(N):
            idx = S_single[j].long()
            bio = self.stage_biomarker_index_flat[idx]
            index_reached[bio] = idx
            IS_normal[bio] = False
            
            abn_mask = ~IS_normal
            if abn_mask.any():
                pa = torch.prod(self.prob_score_gpu[:, index_reached[abn_mask]], dim=1)
            else:
                pa = torch.ones(M, device=self.device, dtype=self.dtype)
            
            if IS_normal.any():
                pn = torch.prod(self.prob_nl_gpu[:, IS_normal], dim=1)
            else:
                pn = torch.ones(M, device=self.device, dtype=self.dtype)
            
            p_perm_k[:, j + 1] = coeff * pa * pn
        
        return p_perm_k

    def _single_mixture_loglike(self, S: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        """
        Compute log-likelihood for a single (S, f) pair. Used with vmap.
        
        Args:
            S: (N_S, N) sequence matrix for all subtypes
            f: (N_S,) fractions
            
        Returns:
            scalar log-likelihood
        """
        N_S = S.shape[0]
        N = S.shape[1]
        M = self.M_subjects
        
        # Compute p_perm_k for each subtype
        p_perm_k = torch.zeros((M, N + 1, N_S), device=self.device, dtype=self.dtype)
        for s in range(N_S):
            p_perm_k[:, :, s] = self._single_sequence_likelihood(S[s])
        
        # Mixture
        f_exp = f.view(1, 1, N_S)
        total_prob_stage = torch.sum(p_perm_k * f_exp, dim=2)
        total_prob_subj = torch.sum(total_prob_stage, dim=1)
        return torch.sum(torch.log(total_prob_subj + 1e-250))

    def _batch_loglike(self, S_batch: torch.Tensor, f_batch: torch.Tensor) -> torch.Tensor:
        """
        Compute log-likelihoods for batch of (S, f) pairs.
        
        Uses a simple loop but with all data on GPU - minimises transfer overhead.
        
        Args:
            S_batch: (K, N_S, N)
            f_batch: (K, N_S)
            
        Returns:
            (K,) log-likelihoods
        """
        K = S_batch.shape[0]
        results = torch.zeros(K, device=self.device, dtype=self.dtype)
        
        # Single GPU kernel per iteration (no CPU-GPU transfer inside loop)
        for k in range(K):
            results[k] = self._single_mixture_loglike(S_batch[k], f_batch[k])
        
        return results

    def _perform_mcmc(self, sustainData, seq_init, f_init, n_iterations, seq_sigma, f_sigma, rng=None):
        """MCMC with batched GPU likelihood."""
        if not self.use_gpu:
            return super()._perform_mcmc(sustainData, seq_init, f_init, n_iterations, seq_sigma, f_sigma, rng)
        
        if rng is None:
            rng = self.global_rng
        
        N = self.stage_score.shape[1]
        N_S = seq_init.shape[0]
        
        BATCH_SIZE = 128  # Larger batch = better GPU utilisation
        
        torch.manual_seed(rng.integers(2**31))
        
        if isinstance(f_sigma, float):
            f_sigma = np.array([f_sigma])
        if isinstance(seq_sigma, (int, float)):
            seq_sigma_np = np.full((N_S, N), seq_sigma)
        else:
            seq_sigma_np = seq_sigma
        
        # Storage
        samples_sequence = np.zeros((N_S, N, n_iterations))
        samples_f = np.zeros((N_S, n_iterations))
        samples_likelihood = np.zeros((n_iterations, 1))
        
        current_sequence = seq_init.copy()
        current_f = f_init.copy().flatten()
        
        # Initial likelihood
        S_t = torch.tensor(current_sequence, device=self.device, dtype=torch.long).unsqueeze(0)
        f_t = torch.tensor(current_f, device=self.device, dtype=self.dtype).unsqueeze(0)
        current_likelihood = self._batch_loglike(S_t, f_t)[0].item()
        
        samples_sequence[:, :, 0] = current_sequence
        samples_f[:, 0] = current_f
        samples_likelihood[0, 0] = current_likelihood
        
        stage_bio = self.stage_biomarker_index.flatten()
        stage_score = self.stage_score.flatten()
        
        iter_idx = 1
        pbar = tqdm(total=n_iterations - 1, desc="MCMC")
        
        while iter_idx < n_iterations:
            batch_end = min(iter_idx + BATCH_SIZE, n_iterations)
            actual_batch = batch_end - iter_idx
            
            # Generate proposals on CPU (constraint logic is complex)
            prop_seqs = np.zeros((actual_batch, N_S, N))
            prop_fs = np.zeros((actual_batch, N_S))
            
            for b in range(actual_batch):
                prop_seq = current_sequence.copy()
                prop_f = current_f.copy()
                
                for s in range(N_S):
                    move_from = rng.integers(0, N)
                    event = int(prop_seq[s, move_from])
                    
                    loc = np.zeros(N, dtype=int)
                    for p in range(N):
                        loc[int(prop_seq[s, p])] = p
                    
                    score = stage_score[event]
                    bio = stage_bio[event]
                    same = stage_bio == bio
                    
                    lower = stage_score[same & (stage_score < score)]
                    upper = stage_score[same & (stage_score > score)]
                    
                    lb = loc[np.where((stage_score == lower.max()) & same)[0][0]] + 1 if len(lower) > 0 else 0
                    ub = loc[np.where((stage_score == upper.min()) & same)[0][0]] if len(upper) > 0 else N
                    
                    if lb < ub:
                        pos = np.arange(lb, ub)
                        w = np.exp(-0.5 * ((pos - move_from) / seq_sigma_np[s, event]) ** 2)
                        w /= w.sum()
                        move_to = rng.choice(pos, p=w)
                        
                        new_seq = np.delete(prop_seq[s], move_from)
                        new_seq = np.insert(new_seq, move_to, event)
                        prop_seq[s] = new_seq
                
                prop_f = np.abs(prop_f + f_sigma * rng.standard_normal(N_S))
                prop_f /= prop_f.sum()
                
                prop_seqs[b] = prop_seq
                prop_fs[b] = prop_f
            
            # BATCHED GPU COMPUTATION - single transfer, many computations
            prop_seqs_t = torch.tensor(prop_seqs, device=self.device, dtype=torch.long)
            prop_fs_t = torch.tensor(prop_fs, device=self.device, dtype=self.dtype)
            batch_likes = self._batch_loglike(prop_seqs_t, prop_fs_t).cpu().numpy()
            
            # Accept/reject on CPU
            for b in range(actual_batch):
                log_ratio = batch_likes[b] - current_likelihood
                if log_ratio > 0 or np.log(rng.random()) < log_ratio:
                    current_sequence = prop_seqs[b].copy()
                    current_f = prop_fs[b].copy()
                    current_likelihood = batch_likes[b]
                
                samples_sequence[:, :, iter_idx] = current_sequence
                samples_f[:, iter_idx] = current_f
                samples_likelihood[iter_idx, 0] = current_likelihood
                iter_idx += 1
                pbar.update(1)
                
                if iter_idx >= n_iterations:
                    break
        
        pbar.close()
        
        ml_idx = np.argmax(samples_likelihood)
        return (
            samples_sequence[:, :, ml_idx],
            samples_f[:, ml_idx],
            samples_likelihood[ml_idx, 0],
            samples_sequence,
            samples_f,
            samples_likelihood
        )

    def _calculate_likelihood(self, sustainData, S: np.ndarray, f: np.ndarray):
        if self.use_gpu:
            try:
                return self.torch_likelihood_calculator.calculate_likelihood(
                    self.torch_sustain_data, S, f
                )
            except RuntimeError as e:
                if "out of memory" in str(e):
                    self.torch_backend.clear_cache()
        return super()._calculate_likelihood(sustainData, S, f)

    def _calculate_likelihood_stage(self, sustainData, S: np.ndarray) -> np.ndarray:
        if self.use_gpu:
            try:
                S_t = torch.tensor(S, device=self.device, dtype=torch.long)
                return self._single_sequence_likelihood(S_t).cpu().numpy()
            except RuntimeError:
                pass
        return super()._calculate_likelihood_stage(sustainData, S)


def create_torch_ordinal_sustain(
    prob_nl, prob_score, score_vals, biomarker_labels,
    N_startpoints=25, N_S_max=3, N_iterations_MCMC=100000,
    output_folder="./output", dataset_name="dataset",
    use_parallel_startpoints=True, seed=None, use_gpu=True, device_id=None
):
    return TorchOrdinalSustain(
        prob_nl, prob_score, score_vals, biomarker_labels,
        N_startpoints, N_S_max, N_iterations_MCMC,
        output_folder, dataset_name, use_parallel_startpoints,
        seed, use_gpu, device_id
    )
