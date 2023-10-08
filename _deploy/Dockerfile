# Use the Debian-based Python 3.11.6 image
FROM python:3.11.6

# Set the working directory inside the container
WORKDIR /app

# Copy the application, server, and its requirements to the container
COPY server.py .
COPY requirements.txt .

# Install system dependencies and Python packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get purge -y --auto-remove gcc && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy the trained model to the container
COPY models/model_colors.pt /app/models/

# Expose the port the app runs on
EXPOSE 8080

# Command to run the Flask application
CMD ["python", "server.py"]