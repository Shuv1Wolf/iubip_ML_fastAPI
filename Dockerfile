FROM python:3.10-slim-buster

# Отключает сохранение кеша питоном
ENV PYTHONDONTWRITEBYTECODE 1
# Если проект крашнется, выведется сообщение из-за какой ошибки это произошло
ENV PYTHONUNBUFFERED 1


WORKDIR /GoodProject/backend/

RUN pip install fastapi
RUN pip install nltk
RUN pip install pymorphy2
RUN pip install uvicorn
RUN pip install Cython --install-option="--no-cython-compile"
RUN pip install -U scikit-learn==1.2.2
RUN pip install pandas

COPY . .