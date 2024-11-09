from juliacall import Main as jl
from juliacall import Pkg as pkg
import numpy as np

pkg.activate("AttentionDCA.jl")
pkg.resolve()
pkg.instantiate()
jl.seval("using AttentionDCA")

def generate_transformer_matrices(fileList, num_epochs=100, H=128, d=5):
    """
    Trains an attention block on different MSAs,
    and returns the Q, K, V matrices for each MSA

    Takes as arguments:
    - fileList: List of str, contains the path to each fasta file containing a MSA of interest
    - num_epochs: int, number of epochs of the attention block training (default: 100)
    - H: int, number of heads of the attention block (default: 128)
    - d: int, dimension of the output space of the Q and K matrices (default: 5)

    Returns:
    - heads_dict: dict, contains the Q, K and V matrices for each MSA
        format: {fileName: {Q: Q_matrices (np.array), K: K_matrices (np.array), V: V_matrices (np.array)}}
    """
    heads_dict = {}
    print("Generation of Q, K, and V matrices")
    for fileName in fileList:
        out_std = jl.trainer("AttentionDCA.jl/data/PF00014.fasta", num_epochs, H=H, d=d)
        V = np.array(out_std.m.V)
        Q = np.array(out_std.m.Q)
        K = np.array(out_std.m.K)
        heads_dict[fileName] = {
            "V": V,
            "Q": Q,
            "K": K,
        }
    
    return heads_dict