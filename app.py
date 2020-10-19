import tensorflow as tf
import streamlit as st
from keras.models import load_model
st.set_option('deprecation.showfileUploaderEncoding', False)
st.cache(allow_output_mutation=True)

MODEL_PATH = 'models/weights-best.h5'

#Load the trained model
model = load_model(MODEL_PATH)


st.write("""
# Malaria Cell Detection Web App
This web application using 5 Convolutional Neural Network (CNN) layer and 2 Fully Connected (FC) layer, using about 26000 blood sample images(Corrected). However this web application cannot guarantee the prediction 100% right, especially for expert case. So the predictions **CANNOT** be used for medical purposes.
""")

st.write(""" The dataset and some information about dataset itself can be found at National Library of Medicine (Index 41). https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7277980/ """)

img_file = st.file_uploader('Please upload your image here', 
                            type=['jpg', 'png'])

from PIL import Image, ImageOps
import numpy as np

def import_and_predict(img_data, model):
    size = (48, 48)
    img = ImageOps.fit(img_data, size, Image.ANTIALIAS)
    x = np.asarray(img)
    x = np.expand_dims(x,axis=0)

    images = np.vstack([x])
    val = model.predict(images)
    return val


if img_file is None:
    st.success('Waiting for your upload')
else:
    image = Image.open(img_file)
    st.image(image, use_column_width=True)
    pred_result = import_and_predict(image, model)
    if pred_result == 0:
        text = 'From this image the result is Parasitized'
    else:
        text = 'From this image the result is Uninfected'
    st.success(text)
    
st.write(""" You can check my Github here https://github.com/YoelRegen """)