import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from functools import partial

class Kernel:
    def __init__(self, kernel_type, gamma):
        self.kernel_type = kernel_type
        self.gamma = gamma
        self.kernel_dict = {
            "rbf": partial(self.rbf_kernel, gamma=self.gamma),
            "laplace": partial(self.laplace_kernel, gamma=self.gamma),
            "linear": self.linear_kernel,
        }

    def rbf_kernel(self, X, Y, gamma=None):
        if gamma is None:
            gamma = 1.0 / X.shape[1]
        
        D = torch.cdist(X, Y, p=2)
        D_squared = torch.pow(D, 2)
        K = torch.exp(-gamma * D_squared)
        return K

    def laplace_kernel(self, X, Y, gamma=1.0):
        D = torch.cdist(X, Y, p=1)
        K = torch.exp(-gamma * D)
        return K

    def linear_kernel(self, X, Y):
        K = torch.matmul(X, Y.transpose(-2, -1))
        return K
    
    def forward(self, X, Y):
        K = self.kernel_dict[self.kernel_type](X, Y)
        return K

class AttentionDCA(nn.Module):
    def __init__(self, reps_matrix, seq_len, num_heads, d_k, d_v, kernel_type, lambda_=0.001, seed=10):
        """
        Args:
            embed_dim (int): Dimension d'entrée et de sortie des embeddings.
            num_heads (int): Nombre de têtes d'attention.
            dropout (float): Taux de dropout appliqué à l'attention.
        """
        super(AttentionDCA, self).__init__()
        self.reps_matrix = reps_matrix
        _, self.input_dim = reps_matrix.shape
        self.num_heads = num_heads
        self.d_k = d_k
        self.lambda_ = lambda_
        self.kernel = Kernel(kernel_type=kernel_type)

        torch.manual_seed(seed)
        self.Q = nn.Parameter(torch.randn(num_heads, seq_len, d_k))
        self.K = nn.Parameter(torch.randn(num_heads, seq_len, d_k))
        self.V_metric = nn.Parameter(torch.randn(num_heads, self.input_dim, d_v))

    def V_aa(self):
        V_aa_1 = torch.matmul(self.reps_matrix, self.V_metric)
        V_aa = self.kernel.forward(V_aa_1, V_aa_1)
        return V_aa
    
    def attention_map_per_head(self, mask=None):
        attention_scores = torch.matmul(self.Q, self.K.transpose(-2, -1))/(self.d_k ** 0.5)

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float("-inf"))

        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_map_per_head = 0.5 * (attention_probs + attention_probs.transpose(-2, -1))
        return attention_map_per_head
    
    def attention_map(self, mask=None):
        attention_map_per_head = self.attention_map_per_head(mask=mask)
        attention_map = torch.sum(attention_map_per_head, dim=0)
        return attention_map
    
    def loss(self, Z, weights, mask=None):
        """
        Compute the loss of the attention model given (Q, K, V), the MSA, and weights of a protein family.

        Args:
            Q (torch.Tensor): Query tensor of shape (H, d, N)
            K (torch.Tensor): Key tensor of shape (H, d, N)
            V (torch.Tensor): Value tensor of shape (H, q, q)
            Z (torch.Tensor): MSA matrix of shape (L, M) with integers 1 to q
            weights (torch.Tensor): Weight vector of shape (M,)
            λ (float): Regularization parameter (default: 0.001)
        
        Returns:
            torch.Tensor: The computed loss value.
        """

        # Compute softmax function of Q * K^T
        attention_map_per_head = self.attention_map_per_head(mask=mask)
        V_aa = self.V_aa()

        # Compute the J tensor
        J = torch.einsum('hij,hqa->ijqa', attention_map_per_head, V_aa)  # Shape: (N, N, q, q)
        mask = ~torch.eye(J.shape[0], dtype=bool, device=J.device).unsqueeze(-1).unsqueeze(-1)  # Exclude self-attention
        J = J * mask

        # Compute the energy of the sequences and the partition function
        L, M = Z.shape  # L: sequence length, M: number of sequences in the MSA
        q = V_aa.shape[1]

        # Create one-hot encoding of Z for indexing
        Z_one_hot = F.one_hot(Z, num_classes=q).permute(2, 0, 1).float()  # Shape: (q, L, M)

        # Compute mat_ene using one-hot encoding for efficient indexing
        mat_ene = torch.einsum('rja,qlm->qrm', J, Z_one_hot)  # Shape: (q, L, M)
        lge = torch.logsumexp(mat_ene, dim=0)  # Shape: (L, M)

        # Compute pl using weights and vectorized operations
        Z_indices = Z.flatten()  # Flatten Z to use for indexing
        mat_ene_flat = mat_ene.view(q, -1)  # Flatten mat_ene for efficient indexing
        mat_ene_correct = mat_ene_flat[Z_indices, torch.arange(L * M)].view(L, M)  # Correct energies
        pl = weights * (mat_ene_correct - lge).sum(dim=0)
        pl = -pl.sum()  # Negate the loss and sum over all sequences

        # Compute the regularization term
        reg = self.lambda_ * torch.sum(J**2)

        # Total loss
        total_loss = pl + reg

        return total_loss

"""
class AttentionDCA(nn.Module):
    def __init__(self, seq_length, num_heads, d_k, num_amino_acids=21, seed=1):
        "Constructor of the model
        parameters :
        seq_length : int, the length of the sequence we are analysing
        num_heads : int, number of heads
        d_k : int, the inner dimension of the model
        num_amino_acids : int, the number of amino acids"
        super(AttentionDCA, self).__init__()
        self.seq_length = seq_length
        self.num_heads = num_heads
        self.d_k = d_k
        self.num_amino_acids = num_amino_acids

        # Initialize Q, K, V matrices
        torch.manual_seed(seed)
        # D'après chat : peut etre amélioré en utilisant une autre initialisation
        self.Q = nn.Parameter(torch.randn(num_heads, seq_length, d_k))
        self.K = nn.Parameter(torch.randn(num_heads, seq_length, d_k))
        self.V = nn.Parameter(torch.randn(num_heads, num_amino_acids, num_amino_acids))

    def forward(self, x):
        "Computes the forward pass
        parameters :
        x : tensor, the MSA that have been encoded"
        # x shape: (batch_size, seq_length)
        batch_size = x.shape[0]

        J = self.compute_j()

        # Calculate energy
        energy = -torch.sum(J[torch.arange(self.seq_length).unsqueeze(1), 
                              torch.arange(self.seq_length).unsqueeze(0), 
                              x.unsqueeze(2), 
                              x.unsqueeze(1)], dim=(1,2))

        return energy

    def pseudo_likelihood(self, x):
        # x shape: (batch_size, seq_length)
        batch_size = x.shape[0]
        
        J = self.compute_j()

        # Calculate pseudo-likelihood
        pl = 0
        for i in range(self.seq_length):
            mask = torch.ones(self.seq_length, dtype=bool)
            mask[i] = False
            
            energies = torch.sum(J[i, mask][:, x[:, mask], :], dim=1)
            pl += torch.sum(energies[torch.arange(batch_size), x[:, i]])
            pl -= torch.sum(torch.logsumexp(energies, dim=1))

        return -pl / (batch_size * self.seq_length)

    def train(self, data, num_epochs=100, lr=0.01):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            loss = self.pseudo_likelihood(data)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    def compute_j(self):
        "Computes the J tensor cf equation (4)"

        # Calculate attention scores
        attention_scores = [torch.matmul(self.Q[i], self.K[i].T)/self.d_k for i in range(self.num_heads)]
        attention_probs = torch.softmax(attention_scores, dim=-1) # pas sur de cette operation

        # Calculate J tensor
        J = torch.tensor([torch.matmul(attention_probs[i], self.V[i]) for i in range(self.num_heads)])
        J = J.sum(dim=0)  # Sum over heads

        return J
"""