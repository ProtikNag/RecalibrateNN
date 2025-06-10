# RecalibrateNN

`RecalibrateNN` is a PyTorch-based project designed to recalibrate a pre-trained neural network (GoogleNet) for binary classification, with a focus on aligning model predictions with a specific concept (e.g., "stripes" for zebra classification) using Concept Activation Vectors (CAVs) and TCAV (Testing with CAVs) scores. The project fine-tunes a specific layer of the model to balance classification accuracy and concept alignment, making it a useful tool for interpretability and controlled model adjustment experiments.

This README provides a step-by-step guide to understanding, setting up, and running the project, as well as instructions for customizing it for your own experiments.

## Project Structure

```
RecalibrateNN/
├── data                   
├── results               
├── .gitignore              
├── config.py               
├── custom_dataloader.py    
├── main.py                 
├── README.md               
├── requirements.txt        
└── utils.py                
```


## Purpose

The project aims to:
1. Fine-tune a pre-trained GoogleNet model for binary classification (e.g., zebra vs. non-zebra).
2. Use CAVs to represent a concept (e.g., "stripes") and align the model's intermediate layer activations with this concept.
3. Evaluate the model's performance using accuracy and TCAV scores before and after recalibration.
4. Provide a framework for experimenting with concept-based model adjustments.

## Prerequisites

- Python 3.8+
- A CUDA-enabled GPU (optional but recommended for faster training; CPU fallback is available)
- Basic familiarity with PyTorch and deep learning concepts

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/RecalibrateNN.git
cd RecalibrateNN
```

### 2. Install Dependencies
Install the required Python packages listed in `requirements.txt`:
```aiignore
pip install -r requirements.txt
```

The key dependencies include:
* `torch` and `torchvision` for model training and image processing
* `numpy` for numerical operations
* `matplotlib` for plotting loss curves
* `scikit-learn` for training CAVs with LinearSVC
* `Pillow` for image loading

### 3. Preparing the Data

The `data/` folder is not included in the repository due to size constraints. You’ll need to set it up manually for binary classification. Here’s how:

Directory Structure
Create the following structure under RecalibrateNN/data/:

```
data/ 
├── binary_classification/
│   ├── class1/
│   │   ├── train/       
│   │   └── valid/       
│   └── class2/          
│       ├── train/       
│       └── valid/       
├── concept/ 
│  ├── stripes_fake/    
│  └── random/         

```

### 4. Verify Configuration

Open config.py and adjust paths or hyperparameters if needed:

* `BINARY_CLASSIFICATION_BASE`: Set to `"./data/binary_classification/"`.
* `CONCEPT_FOLDER`: Set to `"./data/concept/stripes_fake"`.
* `RANDOM_FOLDER`: Set to `"./data/concept/random"`.
* `ZEBRA_CLASS_NAME`: Set to the name of your positive class (e.g., "zebra").
* `RESULTS_PATH`: Where the retrained model will be saved (default: `"./results/retrained_model.pth"`).
* `Hyperparameters` like `LEARNING_RATE`, `EPOCHS`, `BATCH_SIZE`, `LAMBDA_ALIGN`, etc., can be tuned for your experiment.

### 5. Running the Project

Command \
Run the main script:

```bash
#If running in Srikanth platform set the environment variable PLATFORM="Srikanth" using  export PLATFORM="Srikanth"
python main.py
```

### 6. File Details

`config.py`
* Defines hyperparameters, model, and paths.
* Sets random seeds for reproducibility.

``main.py``
* Core script for data loading, CAV computation, training, and evaluation.
* Implements the recalibration logic with a custom loss function.

`custom_dataloader.py`
* MultiClassImageDataset: Loads binary classification images with labels.
* ConceptDataset: Loads concept/random images without labels.

`utils.py`
* get_class_folder_dicts: Parses class folders automatically.
* train_cav: Trains the CAV using LinearSVC.
* cosine_similarity_loss: Computes alignment loss.
* evaluate_accuracy: Calculates model accuracy.

`plot_gradcam_compare.py`
```aiignore
python new_grad.py \
  --model_name vgg \
  --model_before_path ./results/vgg16_before.pth \
  --model_after_path ./results/vgg16_after.pth \
  --target_class zebra \
  --target_layer features.28

```