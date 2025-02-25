# modules/llm_manager.py
import streamlit as st
from langchain_community.llms import Ollama

class LLMManager:
    @staticmethod
    @st.cache_resource
    def get_llm():
        """取得並快取 DeepSeek R1 模型實例"""
        return Ollama(model="deepseek-r1:1.5b")
