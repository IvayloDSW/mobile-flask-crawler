FROM python:3.11-bullseye

# Install system dependencies required by Playwright
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    curl \
    unzip \
    ca-certificates \
    fonts-liberation \
    libnss3 \
    libatk-bridge2.0-0 \
    libxss1 \
    libasound2 \
    libgtk-3-0 \
    libx11-xcb1 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    xdg-utils \
    libgbm1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install playwright

# Copy app code
COPY . .

# Add and allow execution of runtime install script
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Use shell script to install browsers at runtime
CMD ["/app/start.sh"]
