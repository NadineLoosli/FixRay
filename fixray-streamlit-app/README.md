# FixRay Streamlit Application

## Overview
FixRay is a Streamlit application designed for fracture detection in medical images. Users can upload JPG, JPEG, or PNG images, and the application will analyze them using a pre-trained model to identify fractures.

## Project Structure
```
fixray-streamlit-app
├── src
│   ├── app.py                # Main entry point for the Streamlit application
│   ├── pages
│   │   └── upload.py         # Upload functionality for image analysis
│   ├── components
│   │   └── ui.py             # Reusable UI components
│   ├── inference
│   │   ├── __init__.py       # Package initialization for inference
│   │   └── predict_single_image.py  # Functions for model loading and image analysis
│   ├── config.py             # Configuration settings for the application
│   └── utils
│       └── file_helpers.py    # Utility functions for file operations
├── models                     # Directory for storing trained models
├── outputs                    # Directory for storing output images
├── tests
│   └── test_predict.py        # Unit tests for prediction functionality
├── requirements.txt           # Dependencies required to run the application
├── .gitignore                 # Files and directories to ignore by Git
└── README.md                  # Documentation for the project
```

## Installation
To set up the FixRay application, follow these steps:

1. Clone the repository:
   ```
   git clone <repository-url>
   cd fixray-streamlit-app
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage
To run the application, execute the following command:
```
streamlit run src/app.py
```

Once the application is running, you can upload images for analysis. Adjust the confidence threshold using the provided slider to customize the detection sensitivity.

## Testing
To ensure the functionality of the application, unit tests are provided in the `tests` directory. You can run the tests using:
```
pytest tests/test_predict.py
```

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.