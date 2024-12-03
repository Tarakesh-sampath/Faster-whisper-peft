# Use an official Python runtime as a parent image
FROM python:3.10.11-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Install any needed packages specified in requirements.txt 
RUN python -m pip install -U pip
RUN python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
RUN python -m pip install sounddevice transformers scipy
RUN apt-get update \
        && apt-get install portaudio19-dev -y 
RUN apt-get install -y ffmpeg

ADD . .

EXPOSE 5000

CMD gunicorn --bind 0.0.0.0:5000 --access-logfile - --error-logfile - run_app:app