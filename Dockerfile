FROM python:3.11-bullseye

# Install system deps with verbose output
RUN apt-get update && \
    apt-get install -y tesseract-ocr tesseract-ocr-eng && \
    tesseract --version && \
    which tesseract && \
    ls -la /usr/bin/tesseract

# Create working dir
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . /app

# Set env for pytesseract
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata

# Run the app
CMD ["python", "main.py"]