import cv2
import os

# Define the dataset directory structure
dataset_dir = "dataset"
train_dir = os.path.join(dataset_dir, "train")
val_dir = os.path.join(dataset_dir, "val")
train_real_dir = os.path.join(train_dir, "real")
train_spoof_dir = os.path.join(train_dir, "spoof")
val_real_dir = os.path.join(val_dir, "real")
val_spoof_dir = os.path.join(val_dir, "spoof")

# Create the directory structure if it doesn't exist
os.makedirs(train_real_dir, exist_ok=True)
os.makedirs(train_spoof_dir, exist_ok=True)
os.makedirs(val_real_dir, exist_ok=True)
os.makedirs(val_spoof_dir, exist_ok=True)

# Set the mode manually using boolean flags
is_train = False  # Set to True for training, False for validation
is_real = True   # Set to True for real images, False for spoof images

# Frames to skip (set to 1 for every frame)
frame_skip = 1  # Capture every 5th frame

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Select the target directory based on flags
if is_train and is_real:
    target_dir = train_real_dir
    target_key = "train_real"
elif is_train and not is_real:
    target_dir = train_spoof_dir
    target_key = "train_spoof"
elif not is_train and is_real:
    target_dir = val_real_dir
    target_key = "val_real"
else:
    target_dir = val_spoof_dir
    target_key = "val_spoof"

# Start capturing frames
frame_count = 0
image_count = len(os.listdir(target_dir))  # Start from existing image count

print(f"Capturing images for {'train' if is_train else 'validation'} - {'real' if is_real else 'spoof'}.")
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    # Draw rectangles around detected faces and crop them with margin
    for (x, y, w, h) in faces:
        # Margin for cropping
        margin = int(x/5)  # Increase this value for a larger margin

        # Add margin to the face rectangle
        x1 = max(x - margin, 0)
        y1 = max(y - margin, 0)
        x2 = min(x + w + margin, frame.shape[1])
        y2 = min(y + h + margin, frame.shape[0])

        # Draw rectangle with margin
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Crop the face with margin
        face_crop = frame[y1:y2, x1:x2]

        # Save cropped face at intervals
        if frame_count % frame_skip == 0:
            filepath = os.path.join(target_dir, f"{target_key}_{image_count}.jpg")
            cv2.imwrite(filepath, face_crop)
            image_count += 1
            print(f"Saved face to {filepath}")

    frame_count += 1

    # Display the video feed with rectangles
    cv2.imshow("Video Feed - Face Detection", frame)

    # Quit if 'q' is pressed
    key_press = cv2.waitKey(1) & 0xFF
    if key_press == ord('q'):
        print("Exiting.")
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
