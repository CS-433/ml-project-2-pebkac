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
    def __init__(self, L, d, q, num_heads, seed=1):
        """Constructor for the attention model. 
        parameters :
        L : int,sequence length
        d : int, the dimension of the representation
        q : int, the number of categories
         """

        super(MultiHeadAttention, self).__init__()
        
        # Initialize dimensions
        self.d = d # Model's dimension
        self.num_heads = num_heads # Number of attention heads
        self.L = L # Dimension of each head's key, query, and value
        
        # Linear layers for transforming inputs
        """
        Commentaires de Jacques:
        - Les dimensions input et output me semblent étranges, et sont en tout cas inadaptées dans notre contexte
        - Les poids de toutes ces matrices doivent impérativement être initialisés avec une seed pour la reproductibilité
        - Ne pas oublier d'exprimer W_v comme une multiplication de deux matrices (D, d_k) et (d_k, D)
        - La matrice W_o est inutile dans notre contexte
        """
        torch.manual_seed(seed)
        #Q = embedding@W_q dim(embedding[i]) = (1, d) L embedding => dim Q = (L, d) selon le cours 
        self.W_q = [nn.Parameter(d, d) for i in range(num_heads)] # Query transformation

        #K = embedding@W_k => dim K = (L, d)
        self.W_k = [nn.Parameter(d, d) for i in range(num_heads)] # Key transformation

        #V = embedding@W_v => (q,q) selon l'article. Selon le cours dim V = (L, Dv)
        #W_v = W_v_up@W_v_down
        self.W_v_up = [nn.Parameter(d, q) for i in range(num_heads)]
        self.W_v_down = [nn.Parameter(q,d) for i in range(num_heads)] # Value transformation
        
        
    def scaled_dot_product_attention(self, X, Q, K, V, mask=None):
        """calculate the attention score and adds it to the original sample. Q, K, v can be obtained by X@W_
        parameters :
        X : tensor, 
        samples
        Q : tensor, 
        query matrix
        K : tensor, 
        Key matrix
        V : tensor, 
        Value Matrix"""
        # Calculate attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d)
        
        # Apply mask if provided (useful for preventing attention to certain parts like padding)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        # Softmax is applied to obtain attention probabilities
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        # Multiply by values to obtain the final output
        """
        Commentaire de Jacques:
        - Cet output est inadapté dans le contexte
        - Devrait rendre attn_score @ X @ V @ X.T, ou X est l'input du modèle (dimension (L, d), où L est le nombre de d'éléments de la séquence)
        """
        output = torch.matmul(attn_probs, V) @ X @ V @ X.T
        return output
        
    def forward(self, X,mask=None):
        """computes the attention output for a model and a sample"""
        Q = [torch.matmul(X, self.W_q[i]) for i in range(self.num_heads)]
        K = [torch.matmul(X, self.W_k[i]) for i in range(self.num_heads)]
        V = [torch.matmul(torch.matmul(X, self.W_v_up[i]), self.W_v_down) for i in range(self.num_heads)]
        
        # Perform scaled dot-product attention
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        return attn_output
