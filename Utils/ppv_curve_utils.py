import torch
import numpy as np

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def compute_fn(J):
    """
    Computes the Frobenius score for each pair of residues in the sequence

    Takes as arguments:
    - J (torch.Tensor of shape (sequence_length, sequence_length, num_amino_acids, num_amino_acids)):
        J tensor computed from the Q, K, and V matrices of the model
    
    Returns:
    - frob_matrix (torch.Tensor of shape (sequence_length, sequence_length)): 
        The element (i, j) of this matrix is the frobenius score of the pair of residues (i, j)
    """
    frob_matrix = torch.sum(torch.pow(J, 2), (2, 3))
    return frob_matrix

def correct_APC(S):
    """
    Performs an average product correction (APC) to adjust the contact scores

    Takes as arguments:
    - S (torch.Tensor of shape (sequence_length, sequence_length)):
        Matrix containing the contact scores for each pair of residues of the sequence
    
    Returns:
    - S_corrected (torch.Tensor of shape (sequence_length, sequence_length)):
        Matrix containing the corrected contact scores for each pair of residues of the sequence
    """
    N = S.shape[0]
    Si = torch.sum(S, dim=0, keepdim=True)
    Sj = torch.sum(S, dim=1, keepdim=True)
    Sa = torch.sum(S) * (1 - 1 / N)
    S_corrected = S - (Sj @ Si) / Sa
    return S_corrected

def compute_ranking(S, min_dist=6, device=device):
    """
    Extracts sorted residue pairs contact scores from a contact score matrix

    Takes as arguments:
    - S (torch.Tensor of shape (sequence_length, sequence_length)): The element (i, j) of S is the predicted contact score for the residue pair (i, j)

    - min_dist (int): distance in the chain above which the contact between two residues is considered non-trivial (default: min_dist=6)

    - device (torch.device object): Device where the matrices needed for this function are stored 

    Returns:
    - sorted_scores (torch.Tensor of shape (number of residue pairs, 3)):
        Each row has the form (i, j, score), where score is the contact score of pair (i, j)
        The tensor is sorted in descending order with respect to the scores
    """
    N = S.shape[0] 
    i_indices, j_indices = torch.triu_indices(N, N, min_dist).to(device)
    scores = S[j_indices, i_indices]
    scores = torch.stack([i_indices, j_indices, scores], dim=1)
    sorted_indices = torch.argsort(scores[:, -1], descending=True)
    sorted_scores = scores[sorted_indices]
    return sorted_scores

def score(model, min_dist=6, mask=None):
    """
    Computes sorted contact scores from the Q, K and V matrices of a multi-headed attention block model

    Takes as arguments:
    - model (AttentionDCA object): multi-headed attention block model

    - min_dist (int): distance in the chain above which the contact between two residues is considered non-trivial (default: min_dist=6)

    - mask (torch.Tensor of shape (sequence_length, sequence_length)): Mask to apply to the attention map (default: mask=None)

    Returns:
    - sorted_scores (torch.Tensor of shape (number of residue pairs, 3)):
        Each row has the form (i, j, score), where score is the contact score of pair (i, j)
        The tensor is sorted in descending order with respect to the scores
    """
    attention_map_per_head = model.attention_map_per_head(mask=mask)
    V_aa = model.V_aa()
    J = torch.einsum('hij,hqa->ijqa', attention_map_per_head, V_aa) 
    J = 0.5 * (J + J.permute(1, 0, 3, 2))

    frob_matrix = compute_fn(J)
    corrected_frob_matrix = correct_APC(frob_matrix)
    sorted_scores = compute_ranking(corrected_frob_matrix, min_dist)
    return sorted_scores

def compute_residue_pair_dist(filedist):
    """
    Extracts distances between pairs of residues in the sequence from a given file

    Takes as arguments:
    - filedist (str): path to the file containing the pairwise distances between the residues in the sequence
                        The file must have rows of the form (i, j, d_ij) or (i, j, -, d_ij)
    Returns:
    - dist_tensor (torch.Tensor of shape (sequence_length, sequence_length)): 
        The element (i, j) of dist_tensor is the distance between residues i and j 
        If the distance between i and j is not contained in filedist, the distance is replaced by float('inf') 
    """
    d = torch.tensor(np.loadtxt(filedist))

    i_indices = d[:, 0].long()
    j_indices = d[:, 1].long()
    distances = d[:, -1].float()
    max_index = max(i_indices.max().item(), j_indices.max().item()) + 1
    dist_tensor = torch.full((max_index, max_index), float('inf'))
    dist_tensor[i_indices, j_indices] = distances

    return dist_tensor

def compute_referencescore(score, dist, min_dist=6, cutoff=8.0, device=device):
    """
    Compares the scores of residue pairs (i, j) to their distance,
    and computes a reference score enabling to evaluate the ability of the score
    to predict contacts between residues

    Takes as arguments:
    - score (torch.Tensor of shape (number of residue pairs, 3)): Output of the compute_ranking function

    - dist (torch.Tensor of shape sequence_length, sequence_length): Output of the compute_residue_pair_dist function

    - min_dist (int): distance in the chain above which the contact between two residues is considered non-trivial (default: min_dist=6)

    - cutoff (float): distance in space (in Angstrom) under which two residues are considered to be in contact (default: cutoff=8.0)

    - device (torch.device object): Device where the matrices needed for this function are stored 

    Returns:
    - ref_score (torch.Tensor of shape (number of residue pairs, 4)): 
        Each row of this tensor has the form (i, j, plm_score, PPV), 
        where i and j are the indices of the residues,
        plm_score is their associated score from the score tensor given as argument,
        PPV is the ratio 'number of true positives'/'number of positive predictions'
        up to the corresponding score present in the same row
    
    Note:
    - In order for ref_score to be correct, the score tensor given as argument must be sorted
        (the lower the row index in score, the higher the score)
    """
    site_i = score[:, 0].long()
    site_j = score[:, 1].long()
    plm_score = score[:, 2]

    d_ij = dist.to(device)[site_i, site_j]
    valid_mask = (site_j - site_i >= min_dist) & (d_ij != float('inf'))

    site_i = site_i[valid_mask]
    site_j = site_j[valid_mask]
    plm_score = plm_score[valid_mask]
    d_ij = d_ij[valid_mask]

    is_below_cutoff = (d_ij < cutoff).cumsum(dim=0).to(torch.float32)
    ctr_tot = torch.arange(1, len(site_i) + 1, dtype=torch.float32).to(device)
    ratio = is_below_cutoff/ctr_tot

    ref_score = torch.stack([site_i, site_j, plm_score, ratio], dim=1)
    return ref_score

def compute_PPV(sorted_scores, filedist, min_separation=6):
    """
    Computes the PPV curve associated to predicted contact scores

    Takes as arguments:
    - sorted_scores (torch.Tensor of shape (number of residue pairs, 3)):
            Each row has the form (i, j, score), where score is the contact score of pair (i, j)
            The tensor is sorted in descending order with respect to the scores

    - filedist (str): path to the file containing the pairwise distances between the residues in the sequence
                        The file must have rows of the form (i, j, d_ij) or (i, j, -, d_ij)
    
    Returns:
    - PPV_curve (torch.Tensor of shape (num residue pairs,)): PPV curve associated to the contact score predictions
    """
    dist = compute_residue_pair_dist(filedist)
    referencescore = compute_referencescore(sorted_scores, dist, min_dist=min_separation)
    PPV_curve = referencescore[:, 3]
    return PPV_curve

def compute_actualPPV(filedist, cutoff=8.0, min_dist=6):
    """
    Computes the best possible PPV curve from the actual distances between residues
    
    Takes as arguments:
    - filedist (str): path to the file containing the pairwise distances between the residues in the sequence
                        The file must have rows of the form (i, j, d_ij) or (i, j, -, d_ij)

    - cutoff (float): distance in space (in Angstrom) under which two residues are considered to be in contact (default: cutoff=8.0)
        min_separation (int): Séparation minimale entre les résidus pour les contacts non triviaux. Par défaut 6.

    - min_dist (int): distance in the chain above which the contact between two residues is considered non-trivial (default: min_dist=6)
    
    Returns:
    - actual_PPV_curve (torch.Tensor of shape (num residue pairs,)): PPV curve associated to the actual distances between residues
    """
    distances = np.loadtxt(filedist)
    distances = torch.tensor(distances, dtype=torch.float32)
    
    residue_i = distances[:, 0].long()
    residue_j = distances[:, 1].long()
    dist_values = distances[:, -1] 
    
    within_cutoff = dist_values <= cutoff
    non_trivial_contacts = torch.abs(residue_i - residue_j) > min_dist
    valid_contacts = within_cutoff & non_trivial_contacts
    l = valid_contacts.sum().item()
    
    total_lines = distances.size(0)
    trivial_contacts = (within_cutoff & ~non_trivial_contacts).sum().item()
    
    x = torch.ones(l)
    scra = torch.tensor([l / x for x in range(l + 1, total_lines - trivial_contacts + 1)], dtype=torch.float32)
    actual_PPV_curve = torch.cat((x, scra))
    
    return actual_PPV_curve