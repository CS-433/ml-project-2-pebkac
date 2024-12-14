"""
This script defines the function used to eaxtract feature importance from a multi-headed attention block
"""

import numpy as np
import torch

def get_feature_importance(model):
    """
    This function extracts feature importance from a multi-headed attention block
    
    Takes as arguments:
    - model (AttentionDCA object defined in Model.AttentionDCA.py): Multi-headed attention block
    
    Returns:
    - mean_feat_importance (np.array of shape (N_features,): Mean feature importance scores across heads of the attention block
    - std_feat_importance (np.array of shape (N_features,): Standard deviation on feature importance scores across heads of the attention block
    """
    W = model.V_metric.cpu().detach()
    A = torch.matmul(W, W.transpose(-2, -1))
    A_list = [A[i, :, :] for i in range(A.shape[0])]
    eigvals_tup, eigvecs_tup = zip(*map(np.linalg.eigh, A_list))
    eigvecs_abs_list = list(map(np.abs, eigvecs_tup))
    weighted_eigvecs_abs_list = [eigvals_tup[i] * eigvecs_abs_list[i] for i in range(len(eigvals_tup))]
    weights_list = [np.sum(item, axis=1) for item in weighted_eigvecs_abs_list]
    relative_weights_list = [item/np.sum(item) for item in weights_list]
    relative_weights_array = np.row_stack(relative_weights_list)
    mean_feat_importance = np.mean(relative_weights_array, axis=0)
    std_feat_importance = np.std(relative_weights_array, axis=0)
    return mean_feat_importance, std_feat_importance
