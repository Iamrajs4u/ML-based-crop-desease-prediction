from flask import Flask, render_template, request
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

# Load the model
model = load_model("model/model.h5")
print("Model loaded successfully")

# Function to predict disease
def pred_tomato_dieas(tomato_plant):
    test_image = load_img(tomato_plant, target_size=(128, 128))
    test_image = img_to_array(test_image) / 255
    test_image = np.expand_dims(test_image, axis=0)
    result = model.predict(test_image)
    pred = np.argmax(result, axis=1)
    return pred[0]

# Create Flask app
app = Flask(__name__)

# Home route
@app.route("/", methods=['GET'])
def home():
    return render_template("index.html")

# Predict route
@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image']
        filename = file.filename
        file_path = os.path.join("static/upload", filename)
        file.save(file_path)
        pred = pred_tomato_dieas(file_path)
        return render_template("result.html", pred_output=pred, user_image=file_path)

# Run the app
if __name__ == "__main__":
    app.run(threaded=False, port=8080)