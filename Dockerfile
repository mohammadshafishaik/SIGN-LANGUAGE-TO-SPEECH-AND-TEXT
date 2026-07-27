FROM python:3.11-slim

# Install system dependencies for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Set environment variables for Hugging Face Spaces
ENV PORT=7860
EXPOSE 7860
ENV PYTHONPATH=/app

# Start the gunicorn server
CMD ["gunicorn", "inference.app_deploy:app", "--bind", "0.0.0.0:7860", "--workers", "2", "--timeout", "120", "--preload"]
