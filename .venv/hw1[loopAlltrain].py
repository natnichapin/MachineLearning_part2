
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
import os  # Make sure to import os to work with files and directories


# Loop

# Load the model
model = load_model("keras_model.h5", compile=False)

# Load the labels
class_names = open("labels2.txt", "r").readlines()

# Create the array of the right shape to feed into the keras model
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Path to the folder containing test images
image_folder_path = "test/test_keras/"

# Loop through each image in the folder
for image_filename in os.listdir(image_folder_path):
    # Construct the full path to the image
    image_path = os.path.join(image_folder_path, image_filename)

    # Skip if it's not an image file (you can add file type checking if needed)
    if not image_filename.lower().endswith(('png', 'jpg', 'jpeg')):
        continue

    try:
        # Load the image
        image = Image.open(image_path).convert("RGB")
    except:
        print(f"Could not open image: {image_filename}")
        continue

    # Resize the image to 224x224 pixels (Teachable Machine expects this size)
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # Convert the image to a numpy array
    image_array = np.asarray(image)

    # Normalize the image (Teachable Machine typically uses this normalization method)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the data array
    data[0] = normalized_image_array

    # Make a prediction using the model
    prediction = model.predict(data)

    # Find the class with the highest prediction value
    index = np.argmax(prediction)
    class_name = class_names[index].strip()  # Strip any newline characters from the label
    confidence_score = prediction[0][index]

    # Print the prediction and confidence score for the current image
    print(f"Image: {image_filename}")
    print(f"Predicted Class: {class_name}")
    print(f"Confidence Score: {confidence_score}\n")
