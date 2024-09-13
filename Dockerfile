# Base image
FROM python:3.10.12-slim

# Install dependencies for building libraries like NumPy
RUN apt-get update && apt-get install -y \
    build-essential \
    gfortran \
    libopenblas-dev \
    liblapack-dev

# Install Python dependencies
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy application code
COPY . .

# Set the entrypoint
CMD ["python", "main.py"]
