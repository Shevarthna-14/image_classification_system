import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import PIL
import requests

# Function to download the model from Google Drive
def download_model_from_drive(url, output_path):
    response = requests.get(url)
    with open(output_path, 'wb') as file:
        file.write(response.content)

# Use your direct download link
drive_url = 'https://drive.google.com/uc?export=download&id=1erWoWn__nPKtaD98WHanbLbrYt61tAZ8'

# Specify the path where you want to save the model
output_path = 'inception_fruits_veg_model.h5'

# Download the model from Google Drive
download_model_from_drive(drive_url, output_path)

# Load the model
model = load_model(output_path)

# Define the image size
IMG_SIZE = (299, 299)

def preprocess_image(img):
    img = img.resize(IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize
    return img_array

# Class names
class_names = ['Apple Braeburn', 'Apple Crimson Snow', 'Apple Golden 1', 'Apple Golden 2', 'Apple Golden 3', 'Apple Granny Smith',
               'Apple Pink Lady', 'Apple Red 1', 'Apple Red 2', 'Apple Red 3', 'Apple Red Delicious', 'Apple Red Yellow 1',
               'Apple Red Yellow 2', 'Apricot', 'Avocado', 'Avocado ripe', 'Banana', 'Banana Lady Finger', 'Banana Red',
               'Beetroot', 'Blueberry', 'Cactus fruit', 'Cantaloupe 1', 'Cantaloupe 2', 'Carambula', 'Cauliflower', 'Cherry 1',
               'Cherry 2', 'Cherry Rainier', 'Cherry Wax Black', 'Cherry Wax Red', 'Cherry Wax Yellow', 'Chestnut', 'Clementine',
               'Cocos', 'Corn', 'Corn Husk', 'Cucumber Ripe', 'Cucumber Ripe 2', 'Dates', 'Eggplant', 'Fig', 'Ginger Root',
               'Granadilla', 'Grape Blue', 'Grape Pink', 'Grape White', 'Grape White 2', 'Grape White 3', 'Grape White 4',
               'Grapefruit Pink', 'Grapefruit White', 'Guava', 'Hazelnut', 'Huckleberry', 'Kaki', 'Kiwi', 'Kohlrabi',
               'Kumquats', 'Lemon', 'Lemon Meyer', 'Limes', 'Lychee', 'Mandarine', 'Mango', 'Mango Red', 'Mangostan',
               'Maracuja', 'Melon Piel de Sapo', 'Mulberry', 'Nectarine', 'Nectarine Flat', 'Nut Forest', 'Nut Pecan',
               'Onion Red', 'Onion Red Peeled', 'Onion White', 'Orange', 'Papaya', 'Passion Fruit', 'Peach', 'Peach 2',
               'Peach Flat', 'Pear', 'Pear 2', 'Pear Abate', 'Pear Forelle', 'Pear Kaiser', 'Pear Monster', 'Pear Red',
               'Pear Stone', 'Pear Williams', 'Pepino', 'Pepper Green', 'Pepper Orange', 'Pepper Red', 'Pepper Yellow',
               'Physalis', 'Physalis with Husk', 'Pineapple', 'Pineapple Mini', 'Pitahaya Red', 'Plum', 'Plum 2', 'Plum 3',
               'Pomegranate', 'Pomelo Sweetie', 'Potato Red', 'Potato Red Washed', 'Potato Sweet', 'Potato White', 'Quince',
               'Rambutan', 'Raspberry', 'Redcurrant', 'Salak', 'Strawberry', 'Strawberry Wedge', 'Tamarillo', 'Tangelo',
               'Tomato 1', 'Tomato 2', 'Tomato 3', 'Tomato 4', 'Tomato Cherry Red', 'Tomato Heart', 'Tomato Maroon',
               'Tomato Yellow', 'Tomato not Ripened', 'Walnut', 'Watermelon']

# Custom CSS for background and styling
custom_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Lobster&display=swap');

.stApp {
    background-image: url("https://path_to_your_background_image.jpg");
    background-size: cover;
}

h1 {
    font-family: 'Lobster', cursive;
    color: white;
    text-align: center;
    margin-top: 50px;
}

</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# App title
st.markdown("<h1>Fruit and Vegetable Classifier</h1>", unsafe_allow_html=True)

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    img = PIL.Image.open(uploaded_file)
    
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image
    img_array = preprocess_image(img)
    
    # Make prediction
    prediction = model.predict(img_array)
    
    # Get predicted class index
    predicted_class = np.argmax(prediction, axis=1)[0]
    
    # Get the predicted class name
    predicted_label = class_names[predicted_class]
    
    # Display prediction result
    st.write(f"Predicted Class: **{predicted_label}**")
