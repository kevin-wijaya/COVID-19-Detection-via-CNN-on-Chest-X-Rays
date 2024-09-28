FROM python:3.10-slim

WORKDIR /app/client

COPY ./client/.dist /app/client

EXPOSE 8001

CMD [ "python", "-m", "http.server", "8001", "--bind", "0.0.0.0" ]