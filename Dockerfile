# Use a specific Python version (adjust if needed)
FROM python:3.10-slim

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the rest of your project files into the container
COPY . .

# Expose the port your app will run on (e.g., 3000)
EXPOSE 3000

# Define the command to run your app
CMD ["python", "main.py"]
