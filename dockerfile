# Use Python 3.9 slim image as base with specific platform
FROM --platform=linux/amd64 python:3.12.8-slim

# Set working directory
WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .
COPY .env /app/.env

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY ./backend /app/backend
# COPY ./frontend /app/frontend
COPY ./agents /app/agents
COPY ./main.py /app/main.py

# Expose port
EXPOSE 8080

# Run FastAPI application
# Command to run the application
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8080"]