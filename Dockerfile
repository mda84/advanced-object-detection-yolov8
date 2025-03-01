# Use an official Python slim image as the base
FROM python:3.9-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy requirements and install them
COPY requirements.txt /app/
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy project files
COPY . /app/

# Expose ports for Gradio (7860) and REST API (8000)
EXPOSE 7860 8000

# Default command: launch the detection script in Gradio mode
CMD ["python", "detect.py", "--mode", "gradio"]
