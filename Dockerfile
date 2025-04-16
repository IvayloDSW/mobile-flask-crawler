FROM debian:bullseye

# Install system deps including tesseract and Python with explicit version
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    && apt-get clean \
    && which tesseract \
    && tesseract --version

# Create working dir
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt /app/

# Install Python deps
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of the files
COPY . /app

# Set env for pytesseract
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata
ENV PATH="/usr/bin:${PATH}"

# Modify pytesseract to use absolute path
RUN echo 'import os\nimport pytesseract\npytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"\nprint(f"Set tesseract path to: {pytesseract.pytesseract.tesseract_cmd}")' > /app/tesseract_config.py

# Run your app with the config
CMD ["sh", "-c", "python3 /app/tesseract_config.py && python3 main.py"]