import torch 
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt

class ForwardDiffusion:
    # Constructor
    def __init__(self, timesteps, startBetaValue, endBetaValue, imageSize):
        # Initialize the values from the input parameters
        self.timesteps = timesteps
        self.betas = self.betaScheduler(startBetaValue, endBetaValue)
        self.imageSize = imageSize

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
    
    # Loading the transformed dataset from google drive
    def load_transformed_dataset(self, train_root, test_root):

        # Basic transformations to all the image applied
        data_transforms = [
            transforms.Resize((self.img_size, self.img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  # Scales data into [0,1]
            transforms.Lambda(lambda t: (t * 2) - 1)  # Scale between [-1, 1]
        ]

        # Compose will put in a single transformation pipeline sequentially
        data_transform = transforms.Compose(data_transforms)

        # Train dataset
        train_dataset = datasets.ImageFolder(
            root=train_root,
            transform=data_transform
        )

        # Test dataset
        test_dataset = datasets.ImageFolder(
            root=test_root,
            transform=data_transform
        )

        # Combine both the datasets
        return data.ConcatDataset([train_dataset, test_dataset])