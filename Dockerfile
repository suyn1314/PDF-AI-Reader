FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip && pip install -r requirements.txt

# ollama deepseek
RUN ollama pull deepseek-r1:1.5b

COPY . .

# Streamlit port（8501）
EXPOSE 8501

# streamlit run app_text.py or app_img.py
#CMD ["streamlit", "run", "app_text.py", "--server.enableCORS", "false"]

CMD ["streamlit", "run", "app_img.py", "--server.fileWatcherType", "none"]

