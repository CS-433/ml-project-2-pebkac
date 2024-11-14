from juliacall import Main as jl
from juliacall import Pkg as pkg
pkg.add("DCAUtils")
jl.seval("using DCAUtils")
import torch
import numpy as np

"""
Note:
These functions use the DCAUtils Julia package because no equivalent Python package was found
The source code of this package can be found on the following GitHub repo: https://github.com/carlobaldassi/DCAUtils.jl
"""

def ReadFasta(file_name, max_gap_fraction, theta, remove_dups=True, verbose=True):
    """
    Reads a FASTA file and computes the alignment matrix as well as the weight of each sequence in the MSA.
    The MSA contains N sequences of length M

    Parameters:
    - file_name (str): The path to the FASTA file containing the sequence alignment

    - max_gap_fraction (float): The maximum allowed fraction of gaps in the alignment
      A sequence with a fraction of of gaps larger than max_gap_fraction is discarded

    - theta (float in [0, 1]): Two sequences are assumed to be similar if the normalized distance between them is smaller than theta * N
      If set to jl.seval(":auto"), theta will automatically be computed with the compute_theta function from DCAUtils

    - remove_dups (bool, optional): flag indicating whether to remove duplicate sequences from the alignment (default: True)

    - verbose (bool, optional): flag that controls the verbosity of the output (default: True)

    Returns:
    - W (torch.Tensor of size (M,)): tensor containing the weight of each sequence. 
      The weight of a sequence is computed as 1/(n * Meff)
      n is the size of the ensemble of similar sequences the sequence of interest belongs to
      Meff is defined as the sum over the values of 1/n for each ensemble of similar sequences

    - Zint (torch.Tensor of size (M, L)): Alignment matrix
      Each row of this matrix corresponds to a sequence
      Each amino acid/gap of the sequence is encoded with integers in {1, 2, ..., 20, 21}

    - N (int): The number of sequences in the alignment

    - M (int): The length of each sequence in the alignment

    - q (int): The maximum state parameter derived from the alignment, which corresponds to the number of different tokens present in the sequence
      (should be equal to 21 when we consider 20 amino acids + 1 gap)

    Raises:
    - ValueError: If the maximum state parameter q exceeds 32
    """
    Z = jl.read_fasta_alignment(file_name, max_gap_fraction)
    
    if remove_dups:
        Z, _ = jl.remove_duplicate_sequences(Z, verbose=verbose)

    N, M = Z.shape
    q = int(np.round(np.max(Z)))

    if q > 32:
        raise ValueError(f"parameter q={q} is too big (max 31 is allowed)")

    W, Meff = jl.compute_weights(Z, q, theta, verbose=verbose)

    W = torch.tensor(W)
    W *= 1.0 / Meff  
    Zint = np.round(Z).astype(int)
    Zint = torch.tensor(Zint)

    return W, Zint, N, M, q

def quickread(file_name, moreinfo=False):
    """
    Reads a FASTA file and computes the alignment matrix as well as the weight of each sequence in the MSA using pre-defined parameters
    (max_gap_fraction = 0.9, theta = jl.seval(":auto"), remove_dups = True, verbose = False)

    Parameters:
    - fileName (str): The path to the FASTA file to be read.

    - moreinfo (bool, optional): If True, returns the number of sequences of the MSA and the sequences length,
      in addition to the alignment matrix and the weights vector (default: False)

    Returns:
    - W (torch.Tensor of size (M,)): tensor containing the weight of each sequence. 
      The weight of a sequence is computed as 1/(n * Meff)
      n is the size of the ensemble of similar sequences the sequence of interest belongs to
      Meff is defined as the sum over the values of 1/n for each ensemble of similar sequences

    - Zint (torch.Tensor of size (M, L)): Alignment matrix
      Each row of this matrix corresponds to a sequence
      Each amino acid/gap of the sequence is encoded with integers in {1, 2, ..., 20, 21}

    - N (int): The number of sequences in the alignment (returned only if moreinfo = True)

    - M (int): The length of each sequence in the alignment (returned only if moreinfo = True)
    """
    Weights, Z, N, M, _ = ReadFasta(file_name, 0.9, jl.seval(":auto"), True, verbose=False)
    
    if moreinfo:
        return Weights, Z, N, M
    return Z, Weights


