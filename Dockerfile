# Use Python base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose port
EXPOSE 8080

# Command to run the Gunicorn server
CMD ["gunicorn", "main:app", "--bind", "0.0.0.0:8080", "--workers", "2"]
