import pytest
from src.inference.predict_single_image import load_model, analyze_image
from src.config import CONFIG

def test_load_model():
    model = load_model(CONFIG['model_path'], CONFIG['device'])
    assert model is not None, "Model should be loaded successfully"

def test_analyze_image_valid():
    # Assuming a valid image path and model are available for testing
    valid_image_path = "path/to/valid/image.jpg"  # Replace with an actual test image path
    model = load_model(CONFIG['model_path'], CONFIG['device'])
    result = analyze_image(valid_image_path, model, CONFIG['output_dir'], confidence_threshold=0.5)
    
    assert result['status'] == 'SUCCESS', "Analysis should be successful"
    assert 'fractures_detected' in result, "Result should contain fractures_detected"
    assert isinstance(result['fractures_detected'], int), "fractures_detected should be an integer"

def test_analyze_image_invalid():
    invalid_image_path = "path/to/invalid/image.jpg"  # Replace with an actual test image path
    model = load_model(CONFIG['model_path'], CONFIG['device'])
    result = analyze_image(invalid_image_path, model, CONFIG['output_dir'], confidence_threshold=0.5)
    
    assert result['status'] == 'FAILURE', "Analysis should fail for invalid image"
    assert 'error_message' in result, "Result should contain an error message"