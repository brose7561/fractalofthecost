# latent_umap_viewer.py
import numpy as np
import umap
import matplotlib.pyplot as plt
import streamlit as st

st.title("Latent Space UMAP Viewer")

# Load latent vectors
st.sidebar.header("Load Latents")
latent_path = st.sidebar.text_input("Latent .npy file", "vae_outputs/latent_umap.npy")

@st.cache_data
def load_latents(path):
    return np.load(path)

if latent_path:
    try:
        z = load_latents(latent_path)
        st.write(f"Loaded latent vectors: {z.shape}")
    except Exception as e:
        st.error(f"Could not load {latent_path}: {e}")
        z = None
else:
    z = None

if z is not None:
    # UMAP controls
    st.sidebar.header("UMAP Parameters")
    n_neighbors = st.sidebar.slider("n_neighbors", 2, 200, 15, step=1)
    min_dist = st.sidebar.slider("min_dist", 0.0, 1.0, 0.1, step=0.01)
    random_state = st.sidebar.number_input("random_state", value=42, step=1)

    if st.sidebar.button("Run UMAP"):
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state
        )
        emb = reducer.fit_transform(z)

        # Plot
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(emb[:, 0], emb[:, 1], s=5, alpha=0.6)
        ax.axis("off")
        st.pyplot(fig)
