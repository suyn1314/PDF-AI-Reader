# modules/embedder_manager.py
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings


class EmbedderManager:
    @staticmethod
    @st.cache_resource
    def get_embedder():
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
