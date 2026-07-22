import streamlit as st
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.preprocessing import preprocess, preview, Modality
import torch
from models.dinov2 import load_dinov2
from models.feature_extractor import extract_embeddings
import numpy as np
from retrieval.faiss_index import load_index
from retrieval.search import search
import rasterio
from PIL import Image


@st.cache_resource
def get_model():
    return load_dinov2()
@st.cache_resource
def get_index(index_path):
    return load_index(index_path)

st.set_page_config(
    page_title="Cross-Modal Satellite Image Retrieval",
    page_icon="🛰️",
    layout="wide",
)

st.title("🛰️ Cross-Modal Satellite Image Retrieval")

st.markdown( 
    "Retrieve similar satellite images using **DINOv2** and **FAISS**."
)

st.divider()

with st.sidebar:

    st.header("⚙️ Search Settings")

    retrieval_mode = st.radio(
        "Retrieval Mode",
        ["Cross Modal", "Same Modal"]
    )

    query_modality = st.radio(
        "Query Modality",
        ["Sentinel-1", "Sentinel-2"]
    )

    query_folder = st.text_input(
        "Dataset Sample Folder"
    )

    k = st.slider(
        "Top-K Results",
        1,
        10,
        5
    )

    retrieve = st.button(
        "🔍 Retrieve",
        use_container_width=True
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.write("""
    Model: DINOv2 Base

    Index: FAISS

    Dataset: BEN-14K

    Supports:
    - Same Modal Retrieval
    - Cross Modal Retrieval
    """)


with st.spinner("Generating embedding and searching..."):
    if not query_folder:
        st.warning("Please enter a dataset sample folder.")
        st.stop()
    if retrieve:
        modality = (
            Modality.SENTINEL_1
            if query_modality == "Sentinel-1"
            else Modality.SENTINEL_2
        )

        image_tensor = preprocess(
            Path(query_folder),
            modality
        )


        query_preview = preview(
            Path(query_folder),
            modality
        )

        st.image(
            query_preview,
            caption="Query Image",
            use_container_width=False,
            width=300
        )


        model, device = get_model()

        image_tensor = image_tensor.unsqueeze(0).to(device)

        embedding = extract_embeddings(
            image_tensor,
            model,
            device
        )

        if retrieval_mode == "Same Modal":

            if query_modality == "Sentinel-1":
                index = get_index("outputs/faiss_index/s1.index")

            else:
                index = get_index("outputs/faiss_index/s2.index")

        else:

            if query_modality == "Sentinel-1":
                index = get_index("outputs/faiss_index/s2.index")

            else:
                index = get_index("outputs/faiss_index/s1.index")

        query_embedding = embedding.cpu().numpy().astype(np.float32)
        distances, indices = search(
            index,
            query_embedding,
            k=k
        )


        if retrieval_mode == "Same Modal":

            if query_modality == "Sentinel-1":
                paths = np.load(
                    "outputs/embeddings/s1_paths.npy",
                    allow_pickle=True
                )

            else:
                paths = np.load(
                    "outputs/embeddings/s2_paths.npy",
                    allow_pickle=True
                )

        else:

            if query_modality == "Sentinel-1":
                paths = np.load(
                    "outputs/embeddings/s2_paths.npy",
                    allow_pickle=True
                )

            else:
                paths = np.load(
                    "outputs/embeddings/s1_paths.npy",
                    allow_pickle=True
                )
        
        retrieved_paths = paths[indices[0]]

        st.subheader("Retrieved Images")
        cols = st.columns(k)
        if retrieval_mode == "Same Modal":
            target_modality = modality
        else:
            target_modality = (
                Modality.SENTINEL_2
                if modality == Modality.SENTINEL_1
                else Modality.SENTINEL_1
            )

        for i, path in enumerate(retrieved_paths):

            preview_image = preview(
                Path(path),
                target_modality
            )

            with cols[i]:
                st.image(
                    preview_image,
                    caption=f"Rank {i+1}",
                    use_container_width=True
                )

                st.caption(f"Distance: {distances[0][i]:.2f}")
