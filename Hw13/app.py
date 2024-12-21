import keras
from keras.utils import img_to_array
from PIL import Image
import numpy as np
import streamlit as st
import pandas as pd

model1 = keras.models.load_model("fashion_mnist_model1.h5")
model2 = keras.models.load_model("fashion_mnist_vgg16_model.h5")

classes = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

def preprocess_image(image):
    image = image.convert('L')
    image = image.resize((28, 28))
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def preprocess_image_vgg16(image):
    image = image.convert('RGB')
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

st.title("Оберіть потрібну модель для передбачень.")
select_model = st.selectbox("Model", ["CNN", "VGG16"])

st.title("Завантажте картинку.")
uploaded_file = st.file_uploader("Виберіть зображення...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Обране зображення', use_column_width=True)

    if st.button("Передбачити"):
        if select_model == "CNN":
            processed_image = preprocess_image(img)
            predictions = model1.predict(processed_image)
            data = {label: pred for label, pred in zip(classes, predictions.flatten())}
            data = {key: [value] for key, value in data.items()}

            df = pd.DataFrame.from_dict(data, orient='index', columns=['Prediction'])
            df['Prediction'] = (df['Prediction'])
            st.write("Вірогідність кожного класу:")
            st.table(df)

            st.write(f"Передбачений клас: {classes[np.argmax(predictions)]}")

        else:
            processed_image = preprocess_image_vgg16(img)
            predictions = model2.predict(processed_image)
            data = {label: pred for label, pred in zip(classes, predictions.flatten())}
            data = {key: [value] for key, value in data.items()}

            df = pd.DataFrame.from_dict(data, orient='index', columns=['Prediction'])
            df['Prediction'] = (df['Prediction'])
            st.write("Вірогідність кожного класу:")
            st.table(df)

            st.write("Передбачений клас.")
            st.write(f"Передбачений клас: {classes[np.argmax(predictions)]}")
