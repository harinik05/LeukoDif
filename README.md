# 🩺 LeukoDif
![image](https://github.com/harinik05/LeukoDif/assets/63025647/2fe5d9ec-d993-4f39-8a32-44c606febe5c)
LeukoDif is a revolutionary AI tool that employs the power of Denoising Diffusion Medical Models (DDMM) to perform image synthesis and segmentation on a Acute Lymphoblastic Leukemia (ALL) dataset. This model has reparametrized VAE architecture, using encoders and decoders as a entry point for latent space in diffusion model. The mathematical formulas & code aspects were adapted from ideas of UC Berkeley paper "Denoising Diffusion Probabilistic Model." 

## Features
-  **Unsupervised machine learning**: Refers to the ability of the ML model to learn from the neural network without having to label the image set. In this particular scenario, we're going to use unsupervised learning for the generation of ALL images, regardless of its type. 
-  **Anomaly Detection**: Detect anomalies or unusual patterns in the propagation of information or behaviors. Sudden deviations from expected diffusion patterns may indicate events and simulate pathological conditions such as outbreaks, cascades, or disruptions in the network.
-  **Robustness to Noise**: Handle noisy data or incomplete information within the network. They are robust to noise because they focus on the overall patterns of information propagation rather than individual data points.
-  **Generation of Ground Truth Data**: In cases where obtaining ground truth labels for medical images is challenging or subjective, image synthesis can be used to generate synthetic ground truth data (maybe not for industrial reasons).

## Dependencies & Set-Up
1. **Python Environment**: Project is written in Python 3 and will require the installation of Python 3.8. Then, use Anaconda to create and activate venv.
2. **Install the dependencies**: Requires the use of various libraries such as GoogleAuth & Torchvision, which can be installed using pip install requirements.txt
3. **Running the program**: To run the code, execute all the blocks of code in Jupyter Notebooks.

## Procedures
### Forward Diffusion Process
The forward diffusion process applies Markov's Chain to each of the timestep, to add Gaussian noise from the aid of Formula 1 (Assets Folder). The training is done through VAE's lower variational bound, wherein the negative log is considered to yield the highest log likelihood possible. The formula is reparametrized through alpha term, wherin the previous timestep noise is not even required to go to the next one. This model employs the usage of PyTorch linspace, for a linear noise scheduler. Through the use of variational bound, this loss is predicted to see if the model essentially comes to a phase of convergence. 

### Backward Diffusion Process
This reverse process involves going in the opposite direction of Markov's Chain
   
Outline of ReadMe Page
1. Cover Image
2. Description and purpose (Introduction)
3. Functionalities
4. Dependencies
5. Thought Process
6. Future Steps
Stable diffusion model for DDMM. Taken from Kaggle dataset: https://www.kaggle.com/datasets/mohammadamireshraghi/blood-cell-cancer-all-4class
