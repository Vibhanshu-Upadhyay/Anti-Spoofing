# Anti-Spoofing Project

This project is designed to detect face spoofing using deep learning techniques. It implements a robust face anti-spoofing system, leveraging a custom-trained convolutional neural network model.

## Getting Started

### Running the Project
To run the project, execute the `testing.py` module. This is the main entry point for testing the anti-spoofing system.

```bash
python testing.py
```

### Model Used
The project utilizes a deep learning model, `face_spoofing_cnn_model_improved4.keras`, trained from scratch for detecting spoofed faces. This model is located in the `models` folder.

### File Structure
- **`testing.py`**: The main module for running the anti-spoofing tests.
- **`models/`**: Contains the trained model file `face_spoofing_cnn_model_improved4.keras`.

## How It Works
The `testing.py` script loads the model and performs inference on input data to classify whether a face is real or spoofed.

## Note
This project was made for learning purpose into the world of computer vision. Make sure the `models/face_spoofing_cnn_model_improved4.keras` file is present in the `models` folder before running the project.


