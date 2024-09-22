
import streamlit as st
from diffusers import StableDiffusionXLPipeline
import torch
from PIL import Image
import io

# Initialize the Stable Diffusion model
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.to("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available

# Apply the LORA model if supported (This is a placeholder for your implementation)
# For actual use, you'll need to apply your LORA model as needed
def apply_lora_model(pipe, lora_model_path):
    # This is a placeholder function. Implement LORA integration here.
    # For example, you might have a method to apply the LORA model to the pipeline.
    pass

# Apply the LORA model (adjust the path as needed)
lora_model_path = ("../../models/Loras/LogoRedmondV2-Logo-LogoRedmAF.safetensors")
apply_lora_model(pipe, lora_model_path)

st.title("Logo Generator")

user_input = st.text_input("Enter your prompt", value="car")
lora_trigger = "logo"
prompt = lora_trigger + user_input

# Adding a slider for the number of images
num_images = st.slider("Select number of images to generate:", 1, 10, 2)

# Adding checkboxes in a single row
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    check1 = st.checkbox("Colorful")
with col2:
    check2 = st.checkbox("B&W")
with col3:
    check3 = st.checkbox("Minimalistic")
with col4:
    check4 = st.checkbox("Detailed")
with col5:
    check5 = st.checkbox("Circle")

# Building the prompt based on checked options
if check1:
    prompt += ", Colorful"
if check2:
    prompt += ", Black and White"
if check3:
    prompt += ", Minimalistic"
if check4:
    prompt += ", Detailed"
if check5:
    prompt += ", Circle"

if st.button("Generate Image"):
    with st.spinner('Generating image...'):
        progress_bar = st.progress(0)

        for i in range(num_images):
            if i % 2 == 0:
                cols = st.columns(2)  # Create two columns only for even index

            # Generate image
            with torch.no_grad():
                generated_image = pipe(prompt).images[0]

            # Convert PIL image to bytes
            buffer = io.BytesIO()
            generated_image.save(buffer, format="PNG")
            img_bytes = buffer.getvalue()

            # Display image in the appropriate column
            with cols[i % 2]:  # Use modulus to toggle between 0 and 1 for column index
                st.image(img_bytes, use_column_width=True)

            # Update progress after each image is generated and displayed
            progress = ((i + 1) / num_images)
            progress_bar.progress(int(progress * 100))

