import torch 
from torch import nn
from torch.nn import functional as F
import math 

class EncoderAttention(nn.Module):
    # Constructor: Initializes the input and output projection layers of linear transformation
    def __init__(self, numberHeads, embeddingDimensionality, inputBias, outputBias):
        super().__init__()
        self.in_proj = nn.Linear(embeddingDimensionality, 3*embeddingDimensionality, bias= inputBias)
        self.out_proj = nn.Linear(embeddingDimensionality, embeddingDimensionality, bias = outputBias)
        self.n_heads = numberHeads
        self.d_head = int(embeddingDimensionality/numberHeads)
    
    # Forward Pass Function
    def forward_pass(self, inputTensor, causalMask):
        # Determine the batch size, sequence, and d_embeded
        batchSize = inputTensor.shape[0]
        sequenceLength = inputTensor.shape[1]
        embeddingDimensionality = inputTensor.shape[2]
        print(f"Batch Size: {batchSize}, sequence length: {sequenceLength}, embeddingDimensionality: {embeddingDimensionality}")

        # Tuple to convert the Q, K, and V tensors into
        reshapedTuple = (batchSize, self.n_heads, sequenceLength, self.d_head)

        # Split the input into 3 equal chunks along the last dimensionality with a size of embeddingDimensionality each
        Q, K, V = self.in_proj(inputTensor).chunk(3,dim=-1)

        Q = Q.view(reshapedTuple)
        K = K.view(reshapedTuple)
        V = V.view(reshapedTuple)

        # Calculate attention scores weights
        weight = Q @ K.transpose(2,3)

        # CausalMask = True, then prevent seeing the future words by filling with -inf
        if causalMask:
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)
        
        # Follow the formula: Attention is all you need article 
        innerClause = weight/ math.sqrt(self.d_head)
        outputAnswer = F.softmax(innerClause, dim=-1)
        attentionOutput = (outputAnswer @ V).transpose(1,2).reshape(inputTensor.shape)
        attentionResult = self.out_proj(attentionOutput)

        return attentionResult