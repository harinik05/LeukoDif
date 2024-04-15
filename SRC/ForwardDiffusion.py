import torch 
import torch.nn.functional as F

class ForwardDiffusion:
    # Constructor
    def __init__(self, timesteps, startBetaValue, endBetaValue):
        # Initialize the values from the input parameters
        self.timesteps = timesteps
        self.betas = self.betaScheduler(startBetaValue, endBetaValue)

        # Initial calculations for terms in the formula
        alphaValue = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(alphaValue, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1,0), value =1)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1-self.alphas_cumprod)
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod )/(1-self.alphas_cumprod_prev)
    
    # Beta scheduler for coming up with ranges of noise adjustment
    def betaScheduler(self, startBetaValue, endBetaValue):
        # Apply linear beta schedule 
        return torch.linspace(startBetaValue, endBetaValue, self.timesteps)

    # Retrieve the index t from a list of values (t = batch dimension)
    def retrieveIndex(self, vals, currentTimestep, xShape):
        sizeOfBatch = currentTimestep.size(0)
        output = vals.gather(-1,currentTimestep.cpu())
        return output.reshape(sizeOfBatch, *((1,) * (len(xShape) - 1))).to(currentTimestep.device)

    #  Work with the forward diffusion process
    def forwardDiffusionEngine(self, originalImageTensor, currentTimestep, device="cpu"):
        # Generate noise of the original image
        noiseOriginal = torch.randn_like(originalImageTensor)
        sqrt_alphas_cumprod_INTHISTIMESTEP = self.retrieveIndex(self.sqrt_alphas_cumprod, currentTimestep, originalImageTensor.shape)
        sqrt_one_minus_alphas_cumprod_INTHISTIMESTEP = self.retrieveIndex(self.sqrt_one_minus_alphas_cumprod, currentTimestep, originalImageTensor.shape)

        # Calculate the mean + variance 
        mean = sqrt_alphas_cumprod_INTHISTIMESTEP.to(device) * originalImageTensor.to(device)
        variance = (sqrt_one_minus_alphas_cumprod_INTHISTIMESTEP *noiseOriginal).to(device)
        return (mean + variance, noiseOriginal.to(device))
    
