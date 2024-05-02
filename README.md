# 🩺 LeukoDif
![image](https://github.com/harinik05/LeukoDif/assets/63025647/2fe5d9ec-d993-4f39-8a32-44c606febe5c)
LeukoDif is a revolutionary AI tool that employs the power of Denoising Diffusion Medical Models (DDMM) to perform image synthesis and segmentation on a Acute Lymphoblastic Leukemia (ALL) dataset. This model has reparametrized VAE architecture, using encoders and decoders as a entry point for latent space in diffusion model. The mathematical formulas & code aspects were adapted from ideas of UC Berkeley paper "Denoising Diffusion Probabilistic Model." 

## 💙 Features
-  **Unsupervised machine learning**: Refers to the ability of the ML model to learn from the neural network without having to label the image set. In this particular scenario, we're going to use unsupervised learning for the generation of ALL images, regardless of its type. 
-  **Anomaly Detection**: Detect anomalies or unusual patterns in the propagation of information or behaviors. Sudden deviations from expected diffusion patterns may indicate events and simulate pathological conditions such as outbreaks, cascades, or disruptions in the network.
-  **Robustness to Noise**: Handle noisy data or incomplete information within the network. They are robust to noise because they focus on the overall patterns of information propagation rather than individual data points.
-  **Generation of Ground Truth Data**: In cases where obtaining ground truth labels for medical images is challenging or subjective, image synthesis can be used to generate synthetic ground truth data (maybe not for industrial reasons).

## 💻 Dependencies & Set-Up
1. **Python Environment**: Project is written in Python 3 and will require the installation of Python 3.8. Then, use Anaconda to create and activate venv.
2. **Install the dependencies**: Requires the use of various libraries such as GoogleAuth & Torchvision, which can be installed using `pip install requirements.txt`
3. **Running the program**: To run the code, execute all the blocks of code in Jupyter Notebooks.

## 🤖 Procedures
### ⏩ Forward Diffusion Process
The forward diffusion process applies Markov's Chain to each of the timestep, to add Gaussian noise from the aid of Formula 1 (Assets Folder). The training is done through VAE's lower variational bound, wherein the negative log is considered to yield the highest log likelihood possible. The formula is reparametrized through alpha term, wherin the previous timestep noise is not even required to go to the next one. This model employs the usage of PyTorch linspace, for a linear noise scheduler. Through the use of variational bound, this loss is predicted to see if the model essentially comes to a phase of convergence. 

### 🔙 Backward Diffusion Process
<img width="585" alt="Screenshot 2024-05-02 at 3 13 58 PM" src="https://github.com/harinik05/LeukoDif/assets/63025647/ed78d468-a770-4104-b1b9-d9c888999f71">

This reverse process involves going in the opposite direction of Markov's Chain, wherin mean + variance is learnt from the neural network. When I pass the actual image to the neural network (UNet), this should be able to predict the noised image and subtract from the current noise to land the denoised work. The segmentation mask can be easily built through a UNet Architecture (as shown in the code from BackwardDiffusion.py). This uses a buch of CV techniques such as positional embeddings, ResNet, self-attention, group normalization, and geLu blocks. 

## 🎱 Future Steps
- Alter to a cosine noise scheduler: Noise will show up gradually in comparison to linear noise scheduler
- Run generative model on semantic way
- Learning covariance matrix - Improves the log likelihood (sample more images)
- Expand the width of UNet
   
## 👓 Citations
[1] https://www.kaggle.com/datasets/mohammadamireshraghi/blood-cell-cancer-all-4class

[2] https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf

