# Face Presentation Attack Detection with Out-of-Distribution (OOD) Detection

This project implements a robust system for face presentation attack detection (anti-spoofing) using PyTorch. It not only classifies images as 'Real' or 'Spoof' but also includes a sophisticated Out-of-Distribution (OOD) detection mechanism to identify **unknown or unseen attack types**.

This is crucial for real-world security, as attackers constantly devise new spoofing methods (e.g., new types of masks, print attacks, or digital replays) that the model may not have been trained on.

## 🌟 Key Features

* **Binary Classification (Model 1):** A `ResNet-50` model is trained to provide a baseline "Real" vs. "Spoof" classifier.
* **OOD Detection (Model 2):** A `ResNet-18` model is used to build a deep feature space.
* **Mahalanobis Distance:** Implements OOD detection using Mahalanobis distance in the feature space. This allows the system to flag inputs that are statistically different from the known training classes, classifying them as "Unknown / Unseen Attack."
    
* **Dataset:** Uses the **CelebA-Spoof** dataset, which is downloaded automatically using the `kagglehub` library.
* **Evaluation:** Includes a comprehensive evaluation pipeline with accuracy, precision, recall, F1-score, and confusion matrices.
* **Inference Script:** Provides a simple-to-use function to test any local image against both the standard classifier and the advanced OOD detector.

## 🛠️ Methodology: OOD with Mahalanobis Distance

While a standard classifier can only predict the classes it was trained on, this project adds a layer of security by quantifying *how similar* a new image is to the training data.

1.  **Feature Extraction:** A `ResNet-18` model is trained on the known classes (e.g., 'Real', 'Print Attack', 'Replay Attack').
2.  **Statistical Modeling:** After training, we pass all training data through the model and extract the deep feature vectors (from the `avgpool` layer). We then compute the class-wise **mean vector ($\mu_c$)** and **covariance matrix ($\Sigma_c$)** for each class $c$. This models each class as a Gaussian distribution in the high-dimensional feature space.
    
3.  **Inference Time:** When a new image is provided:
    * Its feature vector $x$ is extracted.
    * The Mahalanobis distance to each known class $c$ is calculated:
        $$
        D_M(x, c) = \sqrt{(x - \mu_c)^T \Sigma_c^{-1} (x - \mu_c)}
        $$
    * The *minimum* distance among all classes is taken as the in-distribution score.
4.  **Detection:** If this minimum distance is **greater than a predefined `OOD_THRESHOLD`**, the sample is flagged as an "Unknown / Unseen Attack," even if the standard classifier would have confidently (but wrongly) assigned it to a known class.

## 🚀 Getting Started

### 1. Prerequisites

You must have a Kaggle account and your API credentials set up to download the dataset.

1.  Go to your Kaggle account settings (`www.kaggle.com/[YOUR_USERNAME]/account`).
2.  Click `Create New API Token`. This will download a `kaggle.json` file.
3.  Place this file in the required directory.
    * **Linux/macOS:** `~/.kaggle/kaggle.json`
    * **Windows:** `C:\Users\[YOUR_USERNAME]\.kaggle\kaggle.json`

### 2. Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```
2.  Install the required Python libraries. It is highly recommended to use a virtual environment.
    ```bash
    pip install torch torchvision pandas numpy scipy scikit-learn matplotlib tqdm kagglehub Pillow
    ```

### 3. Configuration

Before running, you **must** edit `face_spoof_detector.py` and update one variable:

```python
# ‼️ === IMPORTANT === ‼️
# Change this to a path on your local computer
YOUR_IMAGE_PATH_TO_TEST = "path/to/your/test/image.jpg"  # 👈 REPLACE THIS
