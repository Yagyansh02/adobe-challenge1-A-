# Multi-stage build for PDF Heading Extraction
FROM python:3.10-slim as builder

# Install system dependencies for PDF processing
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /build

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.10-slim

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy Python packages from builder stage
COPY --from=builder /root/.local /root/.local

# Set working directory
WORKDIR /app

# Copy application files
COPY main.py .
COPY combined_heading_extractor.py .
COPY view_final_results.py .

# Copy model directories with their trained models
COPY safe1/ ./safe1/
COPY safe2/ ./safe2/

# Create input/output directories
RUN mkdir -p /app/input /app/output

# Ensure Python can find packages
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app:$PYTHONPATH

# Set entrypoint
ENTRYPOINT ["python", "main.py"]
