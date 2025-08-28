FROM python:3.9-slim

WORKDIR /app

COPY . /app 

RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["fastapi", "run", "main.py", "--port", "5000", "--workers", "1"]