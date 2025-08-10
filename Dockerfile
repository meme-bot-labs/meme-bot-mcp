# Use Python 3.11 slim image for better performance
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    fonts-dejavu-core \
    fonts-liberation \
    fontconfig \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Test imports to catch issues early
RUN python test_imports.py

# Make entrypoint executable
RUN chmod +x docker-entrypoint.sh

# Create a non-root user for security
RUN useradd -m -u 1000 memebot && chown -R memebot:memebot /app
USER memebot

# Expose the port (Railway will override this with PORT env var)
EXPOSE 8086

# Health check for Railway - increased start period for startup time
HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=5 \
    CMD curl -f http://localhost:$PORT/health || exit 1

# Set entrypoint
ENTRYPOINT ["./docker-entrypoint.sh"]
