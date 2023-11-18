FROM python:3.10-slim-buster

# Отключает сохранение кеша питоном
ENV PYTHONDONTWRITEBYTECODE 1
# Если проект крашнется, выведется сообщение из-за какой ошибки это произошло
ENV PYTHONUNBUFFERED 1


WORKDIR /GoodProject/backend/

RUN pip install fastapi nltk pymorphy2 uvicorn
RUN pip install pandas
RUN pip install Cython --install-option="--no-cython-compile"
RUN pip install -U scikit-learn==1.2.2

RUN pip install pandas
RUN pip install prometheus_fastapi_instrumentator

COPY . .