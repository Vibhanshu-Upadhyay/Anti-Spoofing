import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)
import seaborn as sns
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

# === Configuration ===
model_path = "C:\\Users\\vibhanshu upadhyay\\anti-spoofing-2\\models\\face_spoofing_cnn_model_improved4.keras"  # Path to your .keras model
val_data_dir = "C:\\Users\\vibhanshu upadhyay\\anti-spoofing-2\\dataset\\val"  # Path to validation directory
class_names = ["real", "spoof"]  # Update this based on your dataset

# === Load Model ===
model = load_model(model_path)

# === Data Preparation ===
datagen = ImageDataGenerator(rescale=1.0 / 255.0)
val_generator = datagen.flow_from_directory(
    val_data_dir,
    target_size=(128, 128),  # Adjust based on your model input size
    batch_size=32,
    class_mode="binary",  # Binary classification: real or spoof
    shuffle=False,
)

# === Make Predictions ===
y_true = val_generator.classes
y_pred_proba = model.predict(val_generator)  # Probabilities
y_pred = np.where(y_pred_proba > 0.5, 1, 0)  # Binary predictions

# === Metrics ===
# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
cm_display = sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig("confusion_matrix.png")
plt.close()

# Classification Report
print("\n=== Classification Report ===")
print(classification_report(y_true, y_pred, target_names=class_names))

# ROC and AUC
fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.savefig("roc_curve.png")
plt.close()

# === Sample Predictions Visualization ===
def plot_sample_predictions(generator, model, class_names, num_samples=5):
    """
    Visualizes a few predictions along with true labels.
    """
    x_batch, y_batch = next(generator)
    predictions = model.predict(x_batch)

    plt.figure(figsize=(12, 8))
    for i in range(num_samples):
        ax = plt.subplot(1, num_samples, i + 1)
        plt.imshow(x_batch[i])
        true_label = class_names[int(y_batch[i])]
        pred_label = class_names[int(predictions[i] > 0.5)]
        plt.title(f"True: {true_label}\nPred: {pred_label}", fontsize=10)
        plt.axis("off")

plot_sample_predictions(val_generator, model, class_names)
