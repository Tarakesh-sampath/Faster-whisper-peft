# Use an official Python runtime as a base image
FROM python:3.10.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
COPY app.py /app

# Install dependencies
RUN python -m pip install --no-cache-dir sounddevice scipy transformers 
RUN python -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124

# Command to run the script
CMD ["python", "app.py"]