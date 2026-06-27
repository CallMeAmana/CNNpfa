# CNN Data Augmentation Project

A comprehensive image processing and augmentation application combining CNN model inference with YOLO-based data augmentation capabilities.

## Project Overview

This project provides tools for:
- **Visualizing CNN model effects** on images
- **YOLO-based image augmentation** with advanced transformations
- **Model inference** using pre-trained CNN models
- **Streamlit-based UI** for easy interaction and visualization

## Project Structure

```
CNNpfa/
├── app.py                      # Main Streamlit application
├── inference_cnn.py            # CNN model inference utilities
├── utils.py                    # Helper functions and utilities
├── requirements.txt            # Python dependencies
├── style.css                   # Main stylesheet
├── style_watlow.css            # Alternative stylesheet
├── models/
│   ├── best_model_v2.h5       # Pre-trained CNN model
│   └── model_metadata.json     # Model configuration and metadata
└── pages/
    ├── 1_Visualiser_Effet.py   # CNN effect visualization page
    └── 2_YOLO_Augmentation.py  # YOLO augmentation page
```

## Installation

### Prerequisites
- Python 3.8 or higher
- Conda or pip for package management

### Setup

1. **Activate the conda environment:**
```bash
conda activate augmentation
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Usage

### Running the Application

Start the Streamlit app:
```bash
streamlit run app.py
```

The application will open in your browser with two main pages:

#### Page 1: Visualiser Effet
Visualize the effects of the CNN model on input images. Apply model transformations and see real-time results.

#### Page 2: YOLO Augmentation
Apply YOLO-based data augmentation techniques to your images. Includes various augmentation strategies for dataset enhancement.

## File Descriptions

- **app.py** - Main Streamlit application entry point with UI layout and routing
- **inference_cnn.py** - CNN model loading and inference functions
- **utils.py** - Utility functions for image processing and data handling
- **models/best_model_v2.h5** - Pre-trained CNN model weights
- **models/model_metadata.json** - Model configuration, hyperparameters, and metadata

## Requirements

See `requirements.txt` for a complete list of dependencies. Key packages include:
- TensorFlow/Keras (CNN model)
- OpenCV (image processing)
- Streamlit (web interface)
- YOLOv* (object detection and augmentation)

## Notes

- Ensure your conda environment is activated before running the application
- The pre-trained model (`best_model_v2.h5`) should remain in the `models/` directory
- Check `model_metadata.json` for model-specific configurations


