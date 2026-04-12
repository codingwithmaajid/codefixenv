# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy all files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port for FastAPI
EXPOSE 8000

# Start server + inference (simple version)
CMD ["uvicorn", "server.server:app", "--host", "0.0.0.0", "--port", "7860"]

