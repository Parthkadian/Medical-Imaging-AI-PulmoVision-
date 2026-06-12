# Use official Python slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed for OpenCV, scikit-image, etc.
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Expose the port Railway will use
EXPOSE $PORT

# Start the FastAPI app from the api/ directory
CMD ["sh", "-c", "cd api && uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
