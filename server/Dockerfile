FROM python:3.10-slim

WORKDIR /app/server

COPY ./server /app/server/

RUN pip install --no-cache-dir -r requirements.txt

ENV FLASK_APP=/app/server/src/app.py:serve