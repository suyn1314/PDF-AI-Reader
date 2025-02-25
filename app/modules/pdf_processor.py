# modules/pdf_processor.py
import os
import concurrent.futures
import pymupdf
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class PDFProcessor:
    @staticmethod
    def load_pdf(file):
        """將上傳的 PDF 儲存為 temp.pdf 並載入文件"""
        with open("temp.pdf", "wb") as f:
            f.write(file.getvalue())
        loader = PyPDFLoader("temp.pdf")
        return loader.load()

    @staticmethod
    def get_first_page_image(file):
        """提取 PDF 第一頁並儲存為圖片"""
        with open("temp.pdf", "wb") as f:
            f.write(file.getvalue())
        doc = pymupdf.open("temp.pdf")
        os.makedirs("static", exist_ok=True)
        pix = doc[0].get_pixmap()
        image_path = "static/first_page.png"
        pix.save(image_path)
        return image_path

    @staticmethod
    def chunk_documents(docs):
        """利用 RecursiveCharacterTextSplitter 將文件切分成塊"""
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        return splitter.split_documents(docs)

    @staticmethod
    def process_pdf(file):
        """使用 ThreadPoolExecutor 異步處理 PDF：載入並切分文件"""
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_docs = executor.submit(PDFProcessor.load_pdf, file)
            docs = future_docs.result()
            future_chunks = executor.submit(PDFProcessor.chunk_documents, docs)
            documents = future_chunks.result()
        return documents
