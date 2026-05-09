import numpy as np
import torch
from transformers.generation import GenerationConfig, LogitsProcessor
import json
import argparse

TokenCollocation = True
parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float)
args = parser.parse_args()
alpha = args.alpha
print("alpha:", alpha)

class DispersionLogitsProcessor(LogitsProcessor):
    def __init__(
            self,
            token_collocation_matrix: np.ndarray,
            ):
        self.token_collocation_matrix = token_collocation_matrix
    
    def __call__(
            self,
            input_ids: torch.LongTensor,
            scores: torch.FloatTensor
            ) -> torch.FloatTensor:
        pre_token_id = input_ids[-1][-1]
        tensor_z = torch.tensor(self.token_collocation_matrix[pre_token_id.item()],dtype=torch.float32).to(scores.device)
        
        tensor_z = tensor_z-tensor_z.min()+1e-6
        
        tensor_z = tensor_z/tensor_z.sum()

        scores = scores - scores.min() + 1e-6
        
        # tensor_z_score = self._min_max_scaling(scores, tensor_z)
        # scores = scores/scores.sum()
        # # tensor_z = torch.nn.functional.softmax(tensor_z, dim=0)
        # scores = scores*alpha+(1-alpha)*tensor_z
        # scores = scores*(alpha+(1-alpha)*tensor_z)
        scores = scores+alpha*scores*tensor_z
        # scores = scores+alpha*tensor_z_score
        return scores
        
    def _cal_token_collocation_dispersion(
            self,
            pre_token_id: torch.Tensor,
            cur_token_id: torch.Tensor
            ) -> float:
        
        pre_cur_token_count = self.token_collocation_matrix[pre_token_id.item()][cur_token_id.item()]
        cur_pre_token_count = self.token_collocation_matrix[cur_token_id.item()][pre_token_id.item()]
        token_collocation_mean = (pre_cur_token_count + cur_pre_token_count)/2
        cal_token_collocation_dispersion = ((pre_cur_token_count-token_collocation_mean)**2 + (cur_pre_token_count-token_collocation_mean)**2)/2
        
        return cal_token_collocation_dispersion
    
    def _min_max_scaling(
        self,
        scores: torch.Tensor,
        tensor_z: torch.Tensor
        ) -> torch.Tensor:
        
        scores_min = torch.min(scores)
        scores_max = torch.max(scores)
        # if max(token_collocation_dispersion_list)-min(token_collocation_dispersion_list) == 0:
        #     return top_k_values
        tensor_z_scores = torch.tensor([(x - torch.min(tensor_z))/(torch.max(tensor_z)-torch.min(tensor_z)) * (scores_max-scores_min) + scores_min for x in tensor_z]).to(scores.device)
        # sum_values = (top_k_values + torch.tensor(token_collocation_dispersion_list_scaled).to(top_k_values.device))/2
       
        return tensor_z_scores