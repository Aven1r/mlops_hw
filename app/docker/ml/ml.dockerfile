FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y build-essential

RUN mkdir ml_app

COPY ../../ml ./ml_app/ml

WORKDIR /ml_app

RUN pip install -r ./ml/requirements.txt

CMD uvicorn ml.__main__:app --host 0.0.0.0 --port 8001
