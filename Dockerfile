FROM python:3.11-bullseye

# Install system deps with verbose output
RUN apt-get update

# Create working dir
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . /app


# Run the app
CMD ["python", "main.py"]