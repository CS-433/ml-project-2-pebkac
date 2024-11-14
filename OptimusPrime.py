#############################
#To create a transformer 
#λ
#copier de ce site et a remettre en ordre pour notre model
#https://www.datacamp.com/tutorial/building-a-transformer-with-py-torch?dc_referrer=https%3A%2F%2Fwww.google.com%2F
#############################

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy

class MultiHeadAttention(nn.Module):
    """
    Commentaires de Jacques:
    - Les arguments que devrait prendre cet objet sont la dimension d_k, la dimension de la représentation D, le nombre de heads num_heads, 
    et une seed pour l'initialisation des poids du modèle
    
    - L'argument d_model me semble inadapté dans le contexte
    """
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        # Ensure that the model dimension (d_model) is divisible by the number of heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # Initialize dimensions
        self.d_model = d_model # Model's dimension
        self.num_heads = num_heads # Number of attention heads
        self.d_k = d_model // num_heads # Dimension of each head's key, query, and value
        
        # Linear layers for transforming inputs
        """
        Commentaires de Jacques:
        - Les dimensions input et output me semblent étranges, et sont en tout cas inadaptées dans notre contexte
        - Les poids de toutes ces matrices doivent impérativement être initialisés avec une seed pour la reproductibilité
        - Ne pas oublier d'exprimer W_v comme une multiplication de deux matrices (D, d_k) et (d_k, D)
        - La matrice W_o est inutile dans notre contexte
        """
        self.W_q = nn.Linear(d_model, d_model) # Query transformation
        self.W_k = nn.Linear(d_model, d_model) # Key transformation
        self.W_v = nn.Linear(d_model, d_model) # Value transformation
        self.W_o = nn.Linear(d_model, d_model) # Output transformation
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Calculate attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided (useful for preventing attention to certain parts like padding)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        # Softmax is applied to obtain attention probabilities
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        # Multiply by values to obtain the final output
        """
        Commentaire de Jacques:
        - Cet output est inadapté dans le contexte
        - Devrait rendre attn_score @ X @ V @ X.T, ou X est l'input du modèle (dimension (M, D), où M est le nombre de d'éléments de la séquence)
        """
        output = torch.matmul(attn_probs, V)
        return output
        
    def split_heads(self, x):
        # Reshape the input to have num_heads for multi-head attention
        """
        Commentaire de Jacques:
        - Je pense que le code serait plus compréhensible si l'on ne met pas toutes les matrices W_q, W_k, W_v dans la même matrice
        - Il serait peut être mieux de faire des listes ou l'on met les matrices W_q, W_k, W_v séparément (ce sera plus compréhensible)
        """
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        # Combine the multiple heads back to original shape
        """
        Commentaire de Jacques:
        - Inadapté dans le contexte
        """
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None):
        # Apply linear transformations and split heads
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        # Perform scaled dot-product attention
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Combine heads and apply output transformation
        """
        Commentaire de Jacques:
        - Inadapté dans le contexte
        """
        output = self.W_o(self.combine_heads(attn_output))
        return output
