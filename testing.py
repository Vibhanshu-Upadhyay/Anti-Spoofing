import cv2
import numpy as np
from keras.models import load_model

# Load the pre-trained anti-spoofing model
model_path = "C:\\Users\\vibhanshu upadhyay\\anti-spoofing-2\\models\\face_spoofing_cnn_model_improved4.keras"
try:
    anti_spoofing_model = load_model(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Define function to preprocess input image
def preprocess_image(img):
    img = cv2.resize(img, (128, 128))  # Resize image to match input size of the model
    img = img.astype('float32') / 255  # Normalize pixel values to between 0 and 1
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Define function to predict if the face is spoofed or real
def predict_spoof(img):
    preprocessed_img = preprocess_image(img)
    prediction = anti_spoofing_model.predict(preprocessed_img)
    return prediction[0][0]  # Return the prediction value (probability of spoof)

# Initialize face detector (Haar Cascade for simplicity; you can use DNN-based detectors like RetinaFace or YOLO for better accuracy)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Couldn't access camera")
        cap.release()
        cv2.destroyAllWindows()
        break

    frame = cv2.flip(frame, 1)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract face region of interest (ROI)
        face_roi = frame[y:y+h, x:x+w]

        # Predict spoofing
        prediction = predict_spoof(face_roi)
        if prediction > 0.5:  # Threshold: >0.5 means spoof
            label = "Spoof Detected"
            color = (0, 0, 255)  # Red for spoof
        else:  # Real face
            label = "Real Face"
            color = (0, 255, 0)  # Green for real

        # Draw rectangle around the face and put the label
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Display the frame
    cv2.imshow('Anti-Spoofing Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
