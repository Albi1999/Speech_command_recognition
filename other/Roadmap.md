# Roadmap

## **1. Task Definition**

- Choose which commands to classify: for example, the standard 10 keywords (`yes`, `no`, `up`, `down`, etc.), possibly including `unknown` and `silence`.
- Check class balance and consider data balancing techniques if necessary.

---

## **2. Initial Modeling (Baseline)**

- Implement a **basic CNN model**, inspired by [Sainath15]:
  - Input: 40x101 log-Mel spectrogram
  - Architecture: 2-3 convolutional layers + fully connected layers
  - Output: multi-class classification

This will serve as **reference baseline** to compare more advanced models.

---

## **3. Experimentation with Advanced Architectures**

### CNN-Based

- Residual CNN (ResNet block)

### RNN-Based

- CNN + BiLSTM to capture temporal dynamics

### Transformer-Based

- CNN + Transformer
- Vision Transformer

### Experimental Models

- Axial Attention Transformer
- Transformers Blocks with DyT instead of Layer Normalization
- ResGAN: Residual CNN as feature extractor, attention to map the features and a GAN stile discriminator as classifier

---

## **4. Evaluation and Visualization**

- Metrics: Accuracy, Precision, Recall, F1 Score
- Confusion matrix
- T-SNE visualization of learned feature representations to show class clustering
- Spectrogram + prediction visualizations (qualitative error analysis)

---

## **5. Report and Presentation**

- Write the report in LaTeX
  Include:
  - Motivation for each architecture
  - Design choices
  - Explanatory figures and model comparisons
  - Final analysis of complexity and memory usage
- Android App for demo

---
