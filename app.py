import streamlit as st 
from dotenv import load_dotenv
import os 
import openai
from diffusers import StableDiffusionPipeline
import torch


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

#function to generate an image using DALL-E

def generate_image_using_openai(input_text):
    response = openai.Image.create(
        prompt = input_text,
        n = 1,
        size = "512x512"
    )
    image_url = response['data'][0]['url']
    return image_url

def generate_image_using_diffusers(input_text):
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    prompt = input_text
    image = pipe(prompt).images[0] 
    return image


st.title("AI Image Generation APP")

choice = st.sidebar.selectbox("Select your choice", ["Home", "DALL-E", "Diffusers"])
if choice == "Home":
    st.title("AI Image Generation")
    with st.expander("About the app"):
        st.write("This app is a tool to generate images using DALL-E and HuggingFace Diffusers")
    

elif choice == "DALL-E":
    st.subheader("Image Generation using DALL-E")
    input_text = st.text_input("Enter your text")
    if input_text is not None:
        if st.button("Generate Image"):
            st.info(input_text) 
            image_url = generate_image_using_openai(input_text)
            st.image(image_url, caption = "Image Generating using DALL-E")

elif choice == "Diffusers":
    st.subheader("Image Diffusion Using Hugginface Diffusers")
    input_text = st.text_input("Enter your text")
    if input_text is not None:
        if st.button("Generate Image"):
            st.info(input_text) 
            image_url = generate_image_using_diffusers(input_text)
            st.image(image_url, caption = "Image Generating using diffusers")