import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# Load the model
model = load_model("model/model.h5")
print("Model loaded successfully")

# Load and preprocess the test image
test_image = cv2.imread("Dataset/test/Tomato__Bacterial_spot (1).JPG")
test_image = cv2.resize(test_image, (128, 128))
test_image = img_to_array(test_image) / 255
test_image = np.expand_dims(test_image, axis=0)

# Predict the class
result = model.predict(test_image)
pred = np.argmax(result, axis=1)
print("Prediction:", pred)

# Map prediction to disease name
disease_classes = {
    0: "Tomato - Bacteria Spot Disease",
    1: "Tomato - Early Blight Disease",
    2: "Tomato - Healthy and Fresh",
    3: "Tomato - Late Blight Disease",
    4: "Tomato - Leaf Mold Disease",
    5: "Tomato - Septoria Leaf Spot Disease",
    6: "Tomato - Target Spot Disease",
    7: "Tomato - Tomoato Yellow Leaf Curl Virus Disease",
    8: "Tomato - Tomato Mosaic Virus Disease",
    9: "Tomato - Two Spotted Spider Mite Disease"
}

print("Predicted Disease:", disease_classes[pred[0]])