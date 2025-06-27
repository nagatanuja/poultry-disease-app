from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

app = Flask(__name__)
model = load_model('poultry_model.h5')
labels = ['Coccidiosis', 'Healthy', 'New Castle Disease', 'Salmonella']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})

    file = request.files['image']
    img = load_img(file, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_label = labels[np.argmax(prediction)]

    return jsonify({'prediction': predicted_label})

if __name__ == '__main__':
    app.run(debug=True)
