import torch 
from torch import nn
from torch.nn import functional as F
from .EncoderAttention import EncoderAttention

# Embedding module for CLIP - inheritance ;)
class CLIPEmbeddingModule(nn.Module):
    # Constructor
    def __init__(self, vocabSize, embeddingDimensionality, numberTokens):
        super().__init__()
        # Take care of generating the vector for token + words position in text
        self.token_embedding = nn.Embedding(vocabSize, embeddingDimensionality)
        self.positional_embedding = nn.Parameter(torch.zeros((numberTokens, embeddingDimensionality)))

    # forward pass
    def forward_pass(self, numberTokens):
        # Return their sum in forward pass
        return self.token_embedding(numberTokens) + self.positional_embedding

#  Layer stack for CLIP
class CLIPTransformerLayer(nn.Module):
    # Constructor
    def __init__(self, numberHeads, embeddingDimensionality):
        super().__init__()
        # Normalization Layer -> 
        self.norm1 = nn.LayerNorm(embeddingDimensionality)
        self.norm2 = nn.LayerNorm(embeddingDimensionality)

        # Implementation of self attention (instance of EncoderAttention)
        self.self_attention = EncoderAttention(numberHeads, embeddingDimensionality, True, True)
        
        # Feed forward - equivalent to Multilayer Perceptron from 3Blue1Brown tut
        self.feed_forward = nn.Sequential(
            nn.Linear(embeddingDimensionality, 4*embeddingDimensionality),
            nn.GELU(), #QuickGeLU
            nn.Linear(4*embeddingDimensionality, embeddingDimensionality)
        )
    
    # forward pass
    def forward_pass(self, inputTensor):
        # Normalization and self-attention (residual connection)
        forward = inputTensor+ self.self_attention(self.norm1(inputTensor))

        # Return the tensor from feed forward
        return forward + self.feed_forward(self.norm2)

# Main Class for CLIP Architecture Code
class CLIPArc(nn.Module):
    # Constructor
    def __init__(self, vocabularySize = 49408, embeddingDimensionality=300, maxLength  = 90, numberHeads = 12, numberLayers = 12):
        super().__init__()
        # Vocab size = size of dataset, max sequence length (tolerance of a model)
        self.embedding = CLIPEmbeddingModule(vocabularySize, embeddingDimensionality, maxLength)

        # Layers
        self.layers = nn.ModuleList([CLIPTransformerLayer(numberHeads,embeddingDimensionality) for _ in range(numberLayers)])

        # Normalization layer
        self.layernorm = nn.LayerNorm(embeddingDimensionality)
    
    # Forward pass function
    def forward_pass(self, inputTokens):
        # input tokens
        inputTokens = inputTokens.long()

        #  Pass through the embedding layer
        embedded = self.embedding(inputTokens)

        # Pass through the transformer layer
        for layer in self.layers:
            embedded = layer(embedded)
        
        # final layer normalization
        output = self.final_layernorm(embedded)

        return output