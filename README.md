<p align="center">
  <img src="other/logo_unipd.png" alt="" height="70"/>
</p>

# Speech Command Recognition: A Deep Learning Approach

</div>

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white&style=for-the-badge" height="25"/>
  <img alt="Latex" src="https://img.shields.io/badge/Latex-008080?style=for-the-badge&logo=latex&logoColor=white&logoSize=auto" height="25"/>
  <img alt="Kotlin" src="https://img.shields.io/badge/Kotlin-7F52FF?style=for-the-badge&logo=kotlin&logoColor=white&logoSize=auto" height="25"/>
  <img alt="Tensorflow" src="https://img.shields.io/badge/tensorflow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white&logoSize=auto" height="25"/>
  <img alt="Keras" src="https://img.shields.io/badge/keras-D00000?style=for-the-badge&logo=keras&logoColor=white&logoSize=auto" height="25"/>
<p>

</div>
<p>
  <img alt="Visual Studio Code" src="https://img.shields.io/badge/Visual Studio Code-007ACC?logo=VisualStudioCode&logoColor=white&style=for-the-badge" height="25"/>
  <img alt="Android Studio" src="https://img.shields.io/badge/Android%20Studio-3DDC84?style=for-the-badge&logo=Android&logoColor=white&logoSize=auto" height="25"/>
  <img alt="Overleaf" src="https://img.shields.io/badge/Overleaf-47A141?style=for-the-badge&logo=overleaf&logoColor=white&logoSize=auto" height="25"/>
  <p>

</div>
<p>
  <img alt="Kaggle" src="https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logoColor=white&logoSize=auto" height="25"/>
  <img alt="Google Colab" src="https://img.shields.io/badge/Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white&logoSize=auto" height="25"/>
  <img alt="Anaconda" src="https://img.shields.io/badge/Anaconda-44A833?style=for-the-badge&logo=anaconda&logoColor=white&logoSize=auto" height="25"/>
  <img alt="Jupyter" src="https://img.shields.io/badge/Jupyter-F37626?logo=Jupyter&logoColor=white&style=for-the-badge" height="25"/>
  <p>

</div>
<p>
  <img alt="Git" src="https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white&logoSize=auto" height="25"/>
  <img alt="GitHub" src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white&logoSize=auto" height="25"/>
</p>

This repository contains the project for the "Machine Learning for Human Data" course. The goal is the development and evaluation of various deep learning models for the task of **Keyword Spotting (KWS)**, also known as speech command recognition. The project culminates in the deployment of the best-performing model into a functional Android application.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Architectures Implemented](#architectures-implemented)
- [Repository Structure](#repository-structure)
- [How to Run the Project](#how-to-run-the-project)
  - [Prerequisites](#prerequisites)
  - [Kaggle Environment Setup](#kaggle-environment-setup)
  - [Running the Notebook](#running-the-notebook)
- [Android Application](#android-application)
- [Results](#results)
- [Project Report](#project-report)

---

## Project Overview

This work presents a systematic and comparative study of diverse deep learning paradigms for KWS on the **Google Speech Commands V2** dataset. The analysis extends beyond mere accuracy, considering a holistic set of metrics - including F1-score, number of parameters, model size, and inference time - to identify the architecture with the best performance-efficiency trade-off.

The project covers the entire machine learning lifecycle:
1.  **Data Preprocessing:** Implementation of a complete pipeline to transform raw audio files into log-Mel spectrograms, including data augmentation techniques like Time Shifting, Noise Addition, and SpecAugment.
2.  **Modeling:** Design and from-scratch implementation of six distinct neural network architectures.
3.  **Rigorous Evaluation:** Creation of a unified evaluation pipeline to compare the models fairly and reproducibly.
4.  **Deployment:** Conversion of the best model to the TensorFlow Lite format and its integration into a functional Android application for on-device inference.

## Architectures Implemented

A wide range of architectures was explored to compare different approaches to audio data modeling:

1.  **Baseline CNN:** A simple convolutional model to establish a performance benchmark.
2.  **Residual CNN (ResNet):** A deep architecture based on residual blocks to overcome the limitations of shallow networks.
3.  **Hybrid CNN + BiLSTM:** A hybrid model combining the local feature extraction of CNNs with the temporal modeling capabilities of RNNs.
4.  **Hybrid CNN + Transformer:** A hybrid architecture that replaces the RNN with a Transformer encoder, leveraging the self-attention mechanism.
5.  **Vision Transformer (ViT):** A pure attention-based model that treats the spectrogram as a sequence of patches, discarding convolutional inductive biases.
6.  **Experimental (ResNet + SE + GAN-inspired):** A novel and original architecture that augments a ResNet backbone with a Squeeze-and-Excitation attention block and a classifier inspired by GAN discriminators.

## Repository Structure

-   `speech-command-recognition-full.ipynb`: The main Jupyter notebook containing all the code for preprocessing, model definitions, training, and evaluation.
-   `/app`: Contains the source code for the Android project (written in Kotlin).
-   `/MLHD_Report`: Contains the LaTeX source files and the final PDF of the project report.
-   `/output`: Contains the plots, confusion matrices, and classification reports generated during the model evaluation process.

## How to Run the Project

The project is designed to be run on **Kaggle** to leverage free GPUs and simplify dependency management.

### Prerequisites

-   A Kaggle account.
-   The Google Speech Commands V2 dataset.

### Kaggle Environment Setup

1.  **Download the Dataset:** Download the dataset from the official link: [Speech Commands V2 Download](http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz)

2.  **Upload the Dataset to Kaggle:**
    -   Decompress the `tar.gz` archive.
    -   Go to your Kaggle profile and create a "New Dataset".
    -   Upload the decompressed dataset folder. This process may take some time.

3.  **Create a Kaggle Notebook:**
    -   Create a "New Notebook" on Kaggle.
    -   **Enable GPU:** In the notebook's settings panel (on the right), select a GPU accelerator (e.g., `GPU P100`).
    -   **Add the Dataset:** Click on "Add data" and find/add the dataset you uploaded in the previous step. The dataset will be available in the notebook at the path `/kaggle/input/<your-dataset-name>/`.

### Running the Notebook

1.  Upload and open the `speech-command-recognition-full.ipynb` file in your Kaggle notebook environment.
2.  Ensure that the `BASE_PATH` variable at the beginning of the notebook points to the correct path of your dataset on Kaggle.
3.  Run all cells sequentially. The notebook will handle:
    -   Preprocessing the dataset (if `FULL_DATASET` is set to `True`).
    -   Defining, training, and saving each model.
    -   Running the comparative evaluation and generating all plots and reports.

## Android Application

The `/app` directory contains an Android application written in Kotlin that demonstrates the on-device usage of the best model.

-   **Model Used:** `Residual Attention GAN`, selected for its excellent performance.
-   **Functionality:** The app records audio from the microphone, preprocesses it to generate a log-Mel spectrogram, and feeds it to the TFLite model for real-time inference, displaying the predictions to the user.
-   **How to Run:** Open the project in Android Studio and run it on an emulator or a physical device. The `.tflite` model and labels file are included in the app's `assets` folder.

## Results

The final ranking and detailed metrics for all models are presented below:

| Model                  | Accuracy (%) | F1-Score | Num. Params | Size (MB) | Inference (ms) | Norm. Score ↑ | Rank Score ↓ |
| ------------------------ |:------------:|:--------:|:-----------:|:---------:|:--------------:|:-------------:|:------------:|
| **Residual Attention GAN** | **95.36**    | **0.947**|   764,643   |   8.92    |     0.163      |     0.793     |     3.00     |
| Residual CNN             |    95.08     |  0.944   |   313,315   |   3.72    |     0.153      |     0.866     |     3.20     |
| Hybrid CNN-BiLSTM        |    94.72     |  0.916   |   148,899   |   1.79    |     0.136      |   **0.902**   |   **2.86**   |
| Hybrid CNN-Transformer   |    93.21     |  0.924   |   216,547   |   2.65    |     0.099      |     0.881     |     3.73     |
| Vision Transformer       |    85.06     |  0.840   |   164,643   |   2.03    |   **0.098**    |     0.568     |     3.93     |
| Baseline CNN             |    74.99     |  0.737   | **32,035**  | **0.41**  |     0.101      |     0.202     |     4.27     |

For a detailed analysis, including trade-off plots, confusion matrices, and error analysis, please refer to the project report.

## Project Report

The complete project report can be found in the `/MLHD_Report` directory. It contains an in-depth discussion of the methodology, results, and conclusions.
