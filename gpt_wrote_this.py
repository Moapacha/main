import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
from pymilvus import Milvus
import faiss

# Load pre-trained ResNet model
model = models.resnet18(pretrained=True)
model.eval()

# Preprocess input image
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    return input_batch

# Embedding function
def get_embedding(image_path):
    input_batch = preprocess_image(image_path)
    with torch.no_grad():
        output = model(input_batch)
    return output.numpy().flatten()

# Milvus connection
milvus_client = Milvus(host='localhost', port='19530')
collection_name = 'image_embeddings'

# Streamlit app
st.title("Image Retrieval App")

# User uploads an image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
    st.write("Classifying...")

    # Get embedding of the uploaded image
    query_embedding = get_embedding(uploaded_file)

    # Search similar images in Milvus
    search_params = {'nprobe': 16}
    query_vector = [query_embedding.tolist()]
    results = milvus_client.search(collection_name=collection_name, query_records=query_vector, top_k=5, params=search_params)

    # Display results
    st.write("Similar Images:")
    for result in results[0]:
        st.image(result.id, caption=f"Similar Image {result.id}", use_column_width=True)

