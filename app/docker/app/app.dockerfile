FROM python:3.11-slim

RUN mkdir /app

COPY ../../app ./app

WORKDIR /app

RUN pip install -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "ui.py", "--server.port=8501", "--server.address=0.0.0.0"]