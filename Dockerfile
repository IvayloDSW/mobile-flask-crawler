FROM python:3.10-slim

# Prevent interactive prompts from Debian
ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies (Tesseract and OpenCV reqs)
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
 && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata

# Run the app
CMD ["python", "main.py"]
