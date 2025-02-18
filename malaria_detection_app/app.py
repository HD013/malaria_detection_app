import streamlit as st
from PIL import Image
from helper import load_trained_model, make_prediction

# Set Title
st.title("🦠 Malaria Cell Detection")
st.markdown("""
This app allows you to predict the presecence of the malaraia (Plasmodium) parasite in cell images through a Convolutional Neural Network (CNN) model architechture. 
**Credits**
- App built in `Python` + `Streamlit` by [D. Henders 013](https://github.com/HD013) 
---
""")
st.write("Upload a cell image to detect malaria.")


# Load Model
model = load_trained_model()

# Upload Image
uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Make Prediction
    prediction = make_prediction(model, image)

    # Display Result
    st.subheader("Prediction Result:")
    if prediction > 0.5:
        st.error(f"🔴 Malaria **Detected** with {1 - prediction * 100:.2f}% confidence")
    else:
        st.success(f"🟢 No Malaria Detected with {(1 - prediction) * 100:.2f}% confidence")

# Logo image
image = Image.open('logo.jpg')

st.image(image, use_column_width=True)
