import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import load_model
from PIL import Image, ImageOps
from sklearn.metrics import confusion_matrix, classification_report

# Load the model
model = load_model("keras_model.h5", compile=False)

# Load class names and remove leading numbers and spaces for consistency
class_names = [line.strip()[line.find(' ') + 1:] for line in
               open("labels2.txt", "r").readlines()]
# Example: if line is "0 FLAMINGO", it becomes "FLAMINGO"

# Path to test images (organized by class)
test_folder = "test/test_keras_by_class/"

# Lists to store actual and predicted labels
y_true = []
y_pred = []

# Loop through each class folder
for class_name in os.listdir(test_folder):
    class_folder = os.path.join(test_folder, class_name)

    if not os.path.isdir(class_folder):  # Skip files, only process folders
        continue

    # Process each image in the class folder
    for image_filename in os.listdir(class_folder):
        image_path = os.path.join(class_folder, image_filename)

        if not image_filename.lower().endswith(('png', 'jpg', 'jpeg')):
            continue

        try:
            # Load and preprocess the image
            image = Image.open(image_path).convert("RGB")
            image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
            image_array = np.asarray(image).astype(np.float32)
            normalized_image_array = (image_array / 127.5) - 1
            data = np.expand_dims(normalized_image_array, axis=0)

            # Predict using the model
            prediction = model.predict(data)
            predicted_index = np.argmax(prediction)
            predicted_label = class_names[predicted_index]

            # Append actual and predicted labels
            y_true.append(class_name)  # Actual label (from folder name)
            y_pred.append(predicted_label)  # Predicted label

            # ✅ Move the print statement here, after `predicted_label` is defined
            print(f"Image: {image_filename}, Actual: {class_name}, Predicted: {predicted_label}")

        except Exception as e:
            print(f"Error processing {image_filename}: {e}")

# Get unique labels from both true and predicted values
unique_labels = sorted(list(set(y_true + y_pred)))

# Generate Confusion Matrix using unique labels
cm = confusion_matrix(y_true, y_pred, labels=unique_labels)

# Convert to DataFrame for better visualization
cm_df = pd.DataFrame(cm, index=unique_labels, columns=unique_labels)

# Plot the Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm_df, annot=True, fmt='d', cmap="Blues", linewidths=0.5)
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix")

plt.show()

# Print classification report using unique labels
print("\nClassification Report:\n",
      classification_report(y_true, y_pred, target_names=unique_labels))  # Use unique_labels here