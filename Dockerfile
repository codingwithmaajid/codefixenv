# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy all files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# the path to be used
ENV PYTHONPATH=/app

# Expose port for FastAPI / Hf spaces
EXPOSE 7860

# Start server + inference (simple version)
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]