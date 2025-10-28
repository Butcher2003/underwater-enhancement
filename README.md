This is a comprehensive project for underwater image enhancement. It includes the implementation of two deep learning models, `AquaMorphicNet` and `AquaMorphicUNet`, along with scripts for data preparation, training, evaluation, and visualization. This README provides a detailed guide to understanding, setting up, and running the project.

## 📖 Table of Contents
* [ Project Overview](#-project-overview)
* [ Features](#-features)
* [Model Architecture](#-model-architecture)
* [ Getting Started](#-getting-started)
* [ Dataset](#-dataset)
* [Usage](#️-usage)
* [Evaluation](#-evaluation)
* [License](#-license)

## Project Overview

This project tackles the challenging task of underwater image enhancement. Due to the physical properties of light in water, underwater images often suffer from color distortion, low contrast, and haze. This project implements and explores deep learning-based approaches to restore the visual quality of these images. Two novel architectures, `AquaMorphicNet` and `AquaMorphicUNet`, are introduced, leveraging the power of Vision Transformers and U-Net-based structures to achieve state-of-the-art results.

The core of this project lies in a dual-branch approach:
1.  **Aesthetics-Enhancement Branch:** This branch focuses on restoring the visual appeal of the image, correcting color casts, and improving overall clarity.
2.  **Physics-Informed Branch:** This branch is guided by the physical model of underwater image formation, estimating key parameters like the transmission map and atmospheric light to achieve a more physically accurate restoration.

By combining these two branches, the models can produce visually pleasing and physically consistent enhanced images.

## Features

*   **Two Novel Architectures:**
    *   `AquaMorphicNet`: A Vision Transformer (ViT) based model that processes the image in patches, capturing global context effectively.
    *   `AquaMorphicUNet`: A U-Net-like architecture with a Transformer bottleneck, combining the spatial detail preservation of CNNs with the contextual understanding of Transformers.
*   **Physics-Guided Restoration:** Incorporates a physics-based model of underwater image degradation into the learning process for more realistic results.
*   **Comprehensive Workflow:** Includes scripts for:
    *   **Data Preparation:** Splitting the dataset into training and testing sets.
    *   **Training:** Flexible training scripts for both models, with support for various loss functions (L1, Perceptual Loss).
    *   **Evaluation:** Quantitative evaluation using standard metrics like PSNR and SSIM, as well as domain-specific metrics like UCIQE and UIQM.
    *   **Visualization:** Tools to visually inspect the model's output, including the enhanced image and the predicted physical parameters.
*   **Advanced Training Techniques:**
    *   **Fine-tuning:** Scripts to fine-tune pre-trained models for improved performance.
    *   **Data Augmentation:** On-the-fly data augmentation to improve model generalization and prevent overfitting.

## Model Architecture

### AquaMorphicNet

`AquaMorphicNet` is a pure Vision Transformer-based model. It divides the input image into a sequence of patches and processes them using a series of Transformer blocks. This allows the model to capture long-range dependencies and global context, which is crucial for tasks like color correction and dehazing. The model also features a parallel `PhysicsBranch` that estimates the transmission map and atmospheric light, guiding the restoration process.

### AquaMorphicUNet

`AquaMorphicUNet` combines the strengths of U-Net and Transformers. The U-Net's encoder-decoder structure with skip connections is excellent at preserving spatial details, which is vital for image restoration. At the bottleneck of the U-Net, a Transformer block is introduced to process the high-level features, allowing the model to learn global relationships. This hybrid approach aims to achieve a balance between local and global feature extraction. The `PhysicsBranch` from `AquaMorphicNet` is also integrated into this architecture.

## 🔧 Getting Started

### Prerequisites

*   Python 3.x
*   PyTorch
*   Torchvision
*   Einops
*   Scikit-image
*   OpenCV-Python
*   tqdm
*   matplotlib

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Butcher2003/underwater-enhancement
    cd underwater-enhancement
    ```

2.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You will need to create a `requirements.txt` file listing the packages mentioned above.)*

## Dataset

This project uses the **UIEB (Underwater Image Enhancement Benchmark)** dataset. The dataset contains pairs of raw underwater images and their corresponding high-quality reference images.

### Dataset Preparation

Before training, the dataset needs to be split into training and testing sets. A script is provided to automate this process. You will need to set the `GDRIVE_UNSPLIT_DATA_PATH` to the directory containing the 'raw-890' and 'reference-890' folders, and `GDRIVE_SPLIT_DATA_PATH` to your desired output directory.

## ⚙️ Usage

The project is structured into several Python scripts and Jupyter/Colab cells that handle different stages of the workflow.

### 1. Data Splitting

Run the provided script to split your UIEB dataset into training and testing sets.

### 2. Training

*   **`train.py` (for AquaMorphicNet):** This script trains the `AquaMorphicNet` model. You can configure hyperparameters such as learning rate, batch size, and number of epochs within the script.
*   **`train_unet.py` (for AquaMorphicUNet):** This script is used to train the `AquaMorphicUNet` model.
*   **`train_perceptual.py`:** This script trains the `AquaMorphicUNet` model using a combination of L1 loss and VGG Perceptual Loss for potentially better visual results.
*   **`train_augmented.py`:** This script trains the `AquaMorphicUNet` with on-the-fly data augmentation to improve robustness.

To start training, simply run the desired training script from your terminal:
```bash
python train.py
```

### 3. Evaluation

After training, you can evaluate your model's performance on the test set. The evaluation scripts will calculate PSNR, SSIM, UCIQE, and UIQM metrics.

### 4. Inference and Visualization

Scripts are provided to perform inference on a single image and visualize the results. This is useful for qualitative analysis of the model's performance. You can see the original image, the enhanced image, and the predicted transmission and atmospheric light maps side-by-side.

## Evaluation

The models are evaluated using both standard image quality assessment metrics and metrics specifically designed for underwater images.

*   **PSNR (Peak Signal-to-Noise Ratio):** Measures the pixel-wise difference between the enhanced image and the ground truth. Higher is better.
*   **SSIM (Structural Similarity Index):** Measures the similarity in structure, luminance, and contrast between the enhanced image and the ground truth. Higher is better.
*   **UCIQE (Underwater Color Image Quality Evaluation):** A non-reference metric that evaluates the quality of underwater images based on chroma, saturation, and lightness. Higher is better.
*   **UIQM (Underwater Image Quality Measure):** Another non-reference metric that assesses colorfulness, sharpness, and contrast. Higher is better.

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
