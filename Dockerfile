FROM debian:bullseye

# Install system deps including tesseract and Python
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    && apt-get clean

# Create working dir
WORKDIR /app

# Copy files
COPY . /app

# Install Python deps
RUN pip3 install --no-cache-dir -r requirements.txt

# Set env for pytesseract
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata

# Run your app
CMD ["python3", "main.py"]
