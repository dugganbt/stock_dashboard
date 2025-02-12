FROM python:3.9-slim

# Install system-level build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    gfortran \
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to the latest version
RUN pip install --upgrade pip

WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project files
COPY . .

# Expose the port (adjust if your app uses a different one)
EXPOSE 3000

# Run the app
CMD ["python", "main.py"]
