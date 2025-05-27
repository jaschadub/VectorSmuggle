
# Base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy files
COPY . /app

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir     langchain     openai     qdrant-client     pinecone-client     faiss-cpu     fpdf

# Default envs
ENV VECTOR_DB=faiss

# Entrypoint
CMD ["/bin/bash"]
