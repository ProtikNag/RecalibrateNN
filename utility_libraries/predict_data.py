import torch
from PIL import Image
import argparse
import os
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import logging
import seaborn as sns
import ast
import torchvision.transforms as transforms

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("audit_log.log"),
        logging.StreamHandler()
    ]
)

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# Load the pre-trained model
def load_model(base_model_path):
    """
    Load the model state dictionary from the specified path.
    """
    print(base_model_path)
    try:
      model = torch.load(base_model_path)
      model.eval()
    except Exception:
      model = torch.load(base_model_path, weights_only=False)
      model.eval()
    return model

# Define the inference function
def predict_from_directory(directory_path, model, class_names):
    logging.info(f"Starting prediction for directory: {directory_path}")
    results = []
    y_true = []
    y_pred = []
    
    # Iterate through subdirectories
    for subdir in os.listdir(directory_path):
        subdir_path = os.path.join(directory_path, subdir)
        if not os.path.isdir(subdir_path):
            logging.warning(f"Skipping non-directory: {subdir_path}")
            continue  # Skip if not a directory
        
        target_class = subdir  # Subdirectory name is the target class
        logging.info(f"Processing class: {target_class}")
        
        # Recursively iterate through subdirectories
        for root, _, files in os.walk(subdir_path):
            for image_file in files:
                image_path = os.path.join(root, image_file)
                if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    logging.warning(f"Skipping non-image file: {image_file}")
                    continue  # Skip non-image files
                
                logging.info(f"Processing image: {image_path}")
                
                # Define the image transformations
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                
                # Load and preprocess the image
                image = Image.open(image_path).convert("RGB")
                image = transform(image)
                image = image.unsqueeze(0).to(device)
                
                # Perform inference
                with torch.no_grad():
                    outputs = model(image)
                    confidence, predicted = torch.max(torch.nn.functional.softmax(outputs, dim=1), 1)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    class_0_probability = round(probabilities[0][0].item(),3)
                    class_1_probability = round(probabilities[0][1].item(),3)
                    class_2_probability = round(probabilities[0][2].item(),3)
                     
                
                predicted_class = class_names[predicted.item()]
                
                # Append the result
                results.append({
                    "Image Path": image_path,
                    "File Name": image_file,
                    "true_class": target_class,
                    "predicted_class": predicted_class,
                    "Prediction Confidence": confidence.item(),
                    "Class0_Prob":class_0_probability,
                    "Class1_Prob":class_1_probability,
                    "Class2_Prob":class_2_probability
                })
                
                # Collect true and predicted labels for statistics
                y_true.append(target_class)
                y_pred.append(predicted_class)
                
    return results, y_true, y_pred

def calculate_statistics(y_true, y_pred, output_dir, output_excel):
    """Calculate and save prediction statistics"""
    file_suffix = output_excel.split('/')[-1].split('.')[0]
    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    logging.info(f"Overall Accuracy: {accuracy:.4f}")
    
    # Get unique class labels
    unique_classes = sorted(list(set(y_true + y_pred)))
    
    # Generate classification report
    report = classification_report(y_true, y_pred, labels=unique_classes, output_dict=True)
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=unique_classes)
    
    # Save statistics to text file
    stats_file = os.path.join(output_dir, f'prediction_statistics_{file_suffix}.txt')
    with open(stats_file, 'w') as f:

        f.write("PREDICTION STATISTICS\n")
        f.write("=" * 50 + "\n\n")
        f.write('--------------------------------------')
        f.write('----------Confusion Matrix------------')
        f.write(str(cm))
        f.write('----------Confusion Matrix------------')

        f.write(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n\n")
        
        f.write("Per-Class Statistics:\n")
        f.write("-" * 30 + "\n")
        for class_name in unique_classes:
            if class_name in report:
                class_report = report[class_name]
                precision = class_report.get('precision', 0)
                recall = class_report.get('recall', 0)
                f1_score = class_report.get('f1-score', 0)
                support = class_report.get('support', 0)
                f.write(f"{class_name}:\n")
                f.write(f"  Precision: {precision:.4f}\n")
                f.write(f"  Recall: {recall:.4f}\n")
                f.write(f"  F1-Score: {f1_score:.4f}\n")
                f.write(f"  Support: {support}\n\n")
 
    
    
    # Create and save confusion matrix plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=unique_classes, yticklabels=unique_classes)
    plt.title(f'Confusion Matrix (Accuracy: {accuracy:.4f})')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    cm_file = os.path.join(output_dir, f'confusion_matrix_{file_suffix}.png')
    plt.savefig(cm_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Statistics saved to {stats_file}")
    logging.info(f"Confusion matrix saved to {cm_file}")
    
    return accuracy, report

if __name__ == "__main__":
    # Specify the model path and class names
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Image classification script")
    parser.add_argument("--base_model_path", type=str, required=True, help="Path to the model file")
    parser.add_argument("--class_names", type=str, required=True, help="Comma-separated list of class names")
    parser.add_argument("--image_folder", type=str, required=True, help="Path to the folder containing images")
    parser.add_argument("--dest_dir", type=str, required=True, help="Path to the destination directory to save results")
    parser.add_argument("--output_excel", type=str, required=True, help="Path to save the output Excel file")
    args = parser.parse_args()
    
    base_model_path = args.base_model_path 
    image_folder = args.image_folder
    dest_dir = args.dest_dir

    output_excel = os.path.join(dest_dir,args.output_excel)

    # Flatten the nested list if necessary
    if len(args.class_names) == 1 and isinstance(args.class_names[0], str) and args.class_names[0].startswith('['):
        args.class_names = ast.literal_eval(args.class_names[0])
    class_names = {}
    temp = args.class_names.split(",")   
    labels = temp
    print(labels)
    class_names = {i: class_name.strip() for i, class_name in enumerate(temp)}

    logging.info("Starting the script")
    logging.info(f"Class names: {class_names}")
    logging.info(f"Image folder: {image_folder}")
    logging.info(f"Output Excel file: {output_excel}")

    # Load the model
    model = load_model(base_model_path)
    
    # Perform predictions
    results, y_true, y_pred = predict_from_directory(image_folder, model, class_names)
    
    # Get unique classes from predictions
    unique_classes = sorted(list(set(y_true + y_pred)))
    report = classification_report(y_true, y_pred, labels=unique_classes, output_dict=True)
    print(report)

    cm = confusion_matrix(y_true, y_pred, labels=unique_classes)
    cm_data = pd.DataFrame(cm)

    # Save results to an Excel file
    df = pd.DataFrame(results)
    df = pd.concat([df, cm_data], ignore_index=True)
    df.to_excel(output_excel, index=False)
    # Also save results to CSV
    output_csv = output_excel.replace('.xlsx', '.csv')
    df.to_csv(output_csv, index=False)    
    logging.info(f"Results saved to {output_excel}")
    
    # Directory where your files are located
    base_dir = '.'
    #target_dir = os.path.join(base_dir, 'predicted')
    target_dir = dest_dir
    os.makedirs(target_dir, exist_ok=True)
    
    # Calculate and save statistics
    accuracy, report = calculate_statistics(y_true, y_pred, target_dir, output_excel)
    

