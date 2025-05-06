# Guideline for Setting Up YOLOE-PE Training on a New Machine

This guide outlines the steps to set up and run the custom `train_pe.py` script on a new machine, such as a cloud GPU instance.

## 1. Environment Setup

*   **NVIDIA Drivers & CUDA:** Ensure compatible NVIDIA drivers and the correct CUDA Toolkit version (check PyTorch requirements: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)) are installed. Cloud providers often pre-configure this.
*   **Python Environment:** Create and activate a virtual environment:
    ```bash
    # Using venv
    python -m venv yoloe_env
    source yoloe_env/bin/activate

    # Using conda
    # conda create -n yoloe_env python=3.10 # Or your preferred version
    # conda activate yoloe_env
    ```
*   **Install PyTorch:** Install PyTorch with CUDA support (use the official command generator):
    ```bash
    # Example for CUDA 11.8
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    # Example for CUDA 12.1
    # pip install torch torchvision torchaudio
    ```
*   **Install Ultralytics:**
    ```bash
    pip install ultralytics
    ```

## 2. Code and Configuration Files

*   **Clone Ultralytics Repository:** This is the recommended way to get the necessary framework code.
    ```bash
    git clone https://github.com/ultralytics/ultralytics.git
    cd ultralytics
    # Or clone your specific fork/version if applicable
    ```
*   **Copy Custom Files:** Transfer your specific training files into the cloned `ultralytics` directory structure.
    *   Place `train_pe.py` (your script) somewhere logical (e.g., `ultralytics/my_training/train_pe.py`).
    *   Place `mydata.yaml` (e.g., `ultralytics/my_training/mydata.yaml`). **Update internal paths (`path`, `train`, `val`)** to reflect the dataset location on the *new machine*.
    *   Place `my_labels.pt` (e.g., `ultralytics/my_training/my_labels.pt`). **Update the `pe_path` variable in `train_pe.py`** to this new location. **Update the `data` variable in `train_pe.py`** to point to the new `mydata.yaml` location.

## 3. Model Weights

*   **Base Model (`yoloe-v8l-seg.pt`):**
    *   Ensure this file is available. The `ultralytics` library might download it automatically to a cache (e.g., `~/.cache/ultralytics/`), or you can download/copy it manually and place it in a known location (e.g., within your `my_training` folder).
    *   If placed manually, you might need to adjust the line in `train_pe.py` to `model = YOLOE("/path/to/your/yoloe-v8l-seg.pt")`.

## 4. Dataset

*   **Transfer Data:** Copy your entire image and label dataset (folders/files referenced in `mydata.yaml`) to the new machine.
*   **Verify Paths:** Double-check that the `path`, `train`, and `val` paths inside `mydata.yaml` are correct for the new machine.

## 5. Adjust Training Script (`train_pe.py`)

*   **Paths:** Verify/Update the paths set for the `data` and `pe_path` variables.
*   **Hyperparameters:** Adjust `batch` (e.g., 64-256) and `workers` (e.g., 8-16) based on the new GPU's capabilities (e.g., RTX 4090, H100).
*   **Device:** Ensure `device="0"` (or the correct GPU index).

## 6. Run Training

*   Navigate to the directory containing your modified `train_pe.py` (e.g., `cd ultralytics/my_training`).
*   Activate your virtual environment (`source ../../yoloe_env/bin/activate` or `conda activate yoloe_env`).
*   Execute the script:
    ```bash
    python train_pe.py
    ```
*   Monitor output and GPU usage (`nvidia-smi`). Results are typically saved in a `runs/train/...` directory relative to where you run the script.

## Required Files Checklist

These are the essential files you need to gather/prepare:

1.  **`train_pe.py`**: Your custom training script.
2.  **`mydata.yaml`**: Dataset configuration file (with paths updated for the new machine).
3.  **Image and Label Dataset**: The actual dataset files referenced in `mydata.yaml`.
4.  **`my_labels.pt`**: Pre-computed text prompt embeddings and class names.
5.  **`yoloe-v8l-seg.pt`**: Base pre-trained model weights.

*(Standard ultralytics config files like `default.yaml` are handled by the cloned repo/package installation).* 