import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class AttentionDCA(nn.Module):
    def __init__(self, seq_length, num_heads, d_k, num_amino_acids=21, seed=1):
        """Constructor of the model
        parameters :
        seq_length : int, the length of the sequence we are analysing
        num_heads : int, number of heads
        d_k : int, the inner dimension of the model
        num_amino_acids : int, the number of amino acids"""
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
        """Computes the forward pass
        parameters :
        x : tensor, the MSA that have been encoded"""
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
        """Computes the J tensor cf equation (4)"""

        # Calculate attention scores
        attention_scores = [torch.matmul(self.Q[i], self.K[i].T)/self.d_k for i in range(self.num_heads)]
        attention_probs = torch.softmax(attention_scores, dim=-1) # pas sur de cette operation

        # Calculate J tensor
        J = torch.tensor([torch.matmul(attention_probs[i], self.V[i]) for i in range(self.num_heads)])
        J = J.sum(dim=0)  # Sum over heads

        return J
