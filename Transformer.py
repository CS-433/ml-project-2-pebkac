import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from functools import partial

class Kernel:
    """
    This object serves to define the kernel functions allowing to compute a kernel matrix for 2 matrices X and Y

    Arguments of the constructor:
    - kernel_type (str): if kernel_type = "rbf", a RBF kernel is used 
                        if kernel_type = "linear", a linear kernel is used
    - gamma (float): Scale parameter for the RBF kernel
                    If set to None, this parameter is set to 1/D, where D is the dimensionality of each feature vector in X or Y
                    (default: None)
    Arguments of the forward method:
    - X (torch.Tensor of shape (H, N, D)): Tensor whose lines are features vector of a set of N samples
                                            In our case, H is the number of heads of the attention block
    - Y (torch.Tensor of shape (H, M, D)): Tensor whose lines are features vector of a set of M samples
                                            In our case, H is the number of heads of the attention block
    Forward method returns:
    - K (torch.Tensor of shape (H, N, M)): Kernel matrix between the samples contained in X and Y
    """
    def __init__(self, kernel_type, gamma=None):
        self.kernel_type = kernel_type
        self.gamma = gamma
        self.kernel_dict = {
            "rbf": self.rbf_kernel,
            "linear": self.linear_kernel,
        }

    def rbf_kernel(self, X, Y):
        if self.gamma is None:
            self.gamma = 1.0 / X.shape[1]
        
        D = torch.cdist(X, Y, p=2)
        D_squared = torch.pow(D, 2)
        K = torch.exp(-self.gamma * D_squared)
        return K

    def linear_kernel(self, X, Y):
        K = torch.matmul(X, Y.transpose(-2, -1))
        return K
    
    def forward(self, X, Y):
        K = self.kernel_dict[self.kernel_type](X, Y)
        return K

class AttentionDCA(nn.Module):
    def __init__(self, reps_matrix, seq_len, num_heads, d_k, d_v, kernel_type, gamma=None, lambda_=1e-3, seed=10):
        """
        This object defines a multi-headed attention block, as well as useful methods related to it
        We define: 
        - q as the number of amino-acids + 1 gap
        - L (seq_len in the script) as the length of each sequence
        - D (input_dim in the script) as the dimensionality of the input representation

        Arguments of the constructor:
        - reps_matrix (torch.Tensor of shape (q, D)): The i-th row of reps_matrix is the molecular representation of the i-th amino-acid
        - seq_len (int): length of each sequence of the MSA
        - num_heads (int): Number of heads of the multi-headed attention block
        - d_k (int): Dimensionality of the output spaces of Q and K
        - d_v (int): Dimensionality of the output space of V_metric 
        - kernel_type (str): Kernel function used to compute the value matrix from V_metric and reps_matrix
                            if kernel_type = "rbf", a RBF kernel is used 
                            if kernel_type = "linear", a linear kernel is used
        - gamma (float): Scale parameter, only useful if kernel_type == "rbf" (default: None)
        - lambda_ (float): Regularization parameter for the loss (default: 1e-3)
        - seed (int): Seed to initialize the learnable parameters

        Attributes:
        - self.reps_matrix (torch.Tensor of shape (q, D)): = reps_matrix
        - self.input_dim (int): Dimensionality of the representation used
        - self.num_heads (int): = num_heads
        - self.d_k (int): = d_k
        - self.d_v (int): = d_v
        - self.kernel (Kernel object): Kernel defined by kernel_type and gamma
        - self.Q (torch.nn.Parameter of shape (num_heads, seq_len, d_k)): Contains the Q matrix of each head (learnable)
        - self.K (torch.nn.Parameter of shape (num_heads, seq_len, d_k)): Contains the K matrix of each head (learnable)
        - self.V_metric (torch.nn.Parameter of shape (num_heads, seq_len, d_v)): Contains the matrix W of each head (learnable), 
            where the value matrix can be expressed as V = Kernel(W, W)
        """
        super(AttentionDCA, self).__init__()
        self.reps_matrix = reps_matrix
        _, self.input_dim = reps_matrix.shape
        self.num_heads = num_heads
        self.d_k = d_k
        self.lambda_ = lambda_
        self.kernel = Kernel(kernel_type=kernel_type, gamma=gamma)

        torch.manual_seed(seed)
        self.Q = nn.Parameter(torch.randn(num_heads, seq_len, d_k))
        self.K = nn.Parameter(torch.randn(num_heads, seq_len, d_k))
        self.V_metric = nn.Parameter(torch.randn(num_heads, self.input_dim, d_v))

    def V_aa(self):
        """
        Computes the value matrix V for the amino-acid representations

        Returns:
        - V_aa (torch.Tensor of shape (q, q)): The value matrix V for the amino-acids
        """
        V_aa_1 = torch.matmul(self.reps_matrix, self.V_metric)
        V_aa = self.kernel.forward(V_aa_1, V_aa_1)
        return V_aa
    
    def attention_map_per_head(self, mask=None):
        """
        Computes the attention map for each head in the multi-headed attention block

        Arguments:
        - mask (torch.Tensor of shape (seq_len, seq_len)): A binary mask where 0 indicates masked positions (default: None)

        Returns:
        - attention_map_per_head (torch.Tensor of shape (num_heads, seq_len, seq_len)): symmetrized attention map for each head
        """
        attention_scores = torch.matmul(self.Q, self.K.transpose(-2, -1))/(self.d_k ** 0.5)

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float("-inf"))

        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_map_per_head = 0.5 * (attention_probs + attention_probs.transpose(-2, -1))
        return attention_map_per_head
    
    def attention_map(self, mask=None):
        """
        Computes the combined attention map across all heads.

        Arguments:
        - mask (torch.Tensor of shape (seq_len, seq_len)): A binary mask where 0 indicates masked positions (default: None)

        Returns:
        - attention_map_per_head (torch.Tensor of shape (seq_len, seq_len)): symmetrized attention map of the overall model
        """
        attention_map_per_head = self.attention_map_per_head(mask=mask)
        attention_map = torch.sum(attention_map_per_head, dim=0)
        return attention_map
    
    def loss(self, Z, weights, mask=None):
        """
        Computes the loss of the attention model given its parameters and a protein family MSA.
        This method evaluates the loss function based on the attention mechanism, the 
        multiple sequence alignment (MSA), and the regularization term. It calculates 
        sequence energies, partition functions, and incorporates regularization to 
        enforce smoothness of the interaction tensor.

        Args:
        - Z (torch.Tensor of shape (seq_len, num_sequences)): MSA matrix (each column is a sequence of the MSA)
        - weights (torch.Tensor of shape (num_sequences,)): Weight vector, each element corresponds to the weight of a given sequence in the MSA
        - mask (torch.Tensor of shape (seq_len, seq_len)): A binary mask where 0 indicates masked positions (default: None)

        Returns:
        - total_loss(torch.Tensor, float scalar): The total loss of the model for the dataset characterized by Z and weights
        """

        attention_map_per_head = self.attention_map_per_head(mask=mask)
        V_aa = self.V_aa()

        # Compute the J tensor
        J = torch.einsum('hij,hqa->ijqa', attention_map_per_head, V_aa)  
        mask = ~torch.eye(J.shape[0], dtype=bool, device=J.device).unsqueeze(-1).unsqueeze(-1) 
        J = J * mask

        L, M = Z.shape
        q = V_aa.shape[1]

        # Compute energy_matrix using one-hot encoding for efficient indexing
        Z_one_hot = F.one_hot(Z, num_classes=q).permute(2, 0, 1).float()
        energy_matrix = torch.einsum('rja,qlm->qrm', J, Z_one_hot) 
        lge = torch.logsumexp(energy_matrix, dim=0) 

        # Computes the pseudo-likelihood (pl)
        Z_indices = Z.flatten()
        energy_matrix_flat = energy_matrix.view(q, -1)
        energy_matrix_correct = energy_matrix_flat[Z_indices, torch.arange(L * M)].view(L, M)
        pl = weights * (energy_matrix_correct - lge).sum(dim=0)
        pl = -pl.sum()

        # Computes the regularization term (reg)
        reg = self.lambda_ * torch.sum(J**2)

        total_loss = pl + reg
        return total_loss