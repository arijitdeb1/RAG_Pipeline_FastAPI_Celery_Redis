FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY ./app /app

# Copy context.pdf (it's in project root, not in app folder)
#COPY context.pdf /app/context.pdf
