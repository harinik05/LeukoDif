import torch 
import torchvision 
import matplotlib.pyplot as plt
from torchvision import transforms

# Data pre-processing tasks
class PreProcessing():
    # Constructor
    def __init__(self):
        pass

    # Showing images from the dataset in a grid
    def revealImagesGrid(self, dataset, sampleCount, columns):
        # Create a new figure for plotting (15 by 15) grid
        plt.figure(figsize=(15,15))

        # Loop through the entire dataset in google drive
        for _, (image, label) in enumerate(dataset):
            # Still not reached the end
            if _ != sampleCount:
                # Create subplot in the corresponding row, coln
                rowCount = int(sampleCount/columns)
                plt.subplot(rowCount+1, columns, _+1)

                # Permute the dimensions so it displays RGB image (channels, height, width) -> (height, width, channels)
                plt.imshow(image.permute(1,2,0))
            else:
                break
    
    # Showing images in a smaller grid
    def revealSmallGrid(self, inputImage):
        reverse_transforms = transforms.Compose([
            transforms.Lambda(lambda t: (t + 1) / 2),
            transforms.Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
            transforms.Lambda(lambda t: t * 255.),
            transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
            transforms.ToPILImage(),
        ])

        # Take first image of batch
        if len(inputImage.shape) == 4:
            image = image[0, :, :, :]

        plt.imshow(reverse_transforms(inputImage))
        plt.show()
        