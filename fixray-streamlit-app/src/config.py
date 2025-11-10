# Configuration settings for the FixRay application

MODEL_PATH = "models/your_model_file.pth"  # Path to the trained model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Device for model inference
CONFIDENCE_THRESHOLD = 0.5  # Default confidence threshold for predictions
OUTPUT_DIR = "outputs/"  # Directory to save output images