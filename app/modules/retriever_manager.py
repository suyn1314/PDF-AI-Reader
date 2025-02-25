# modules/retriever_manager.py
import json
import streamlit as st
from langchain_community.vectorstores import FAISS
from modules.embedder_manager import EmbedderManager

class RetrieverManager:
    @staticmethod
    @st.cache_resource(hash_funcs={list: lambda _: None})
    def create_retriever(_docs):
        """
        將文件列表轉換成 JSON 並建立 FAISS 檢索器。
        _docs: 文件列表，每個文件應具有 page_content 與 metadata 屬性。
        """
        docs_json = json.dumps(
            [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in _docs]
        )
        docs_list = json.loads(docs_json)
        embedder = EmbedderManager.get_embedder()
        vector = FAISS.from_texts(
            [doc["page_content"] for doc in docs_list],
            embedder,
            metadatas=[{"source": doc["metadata"].get("source", "Uploaded PDF")} for doc in docs_list]
        )
        return vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})
