FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the API code
COPY main.py .

# Create directory for models
RUN mkdir -p /app/models

# Copy your pre-trained models (you'll need to include them in your deployment)
# COPY models/mental_health_traditional_model.joblib /app/models/
# COPY models/mental_health_tfidf_vectorizer.joblib /app/models/

# Set environment variables
ENV PORT=8000

# Expose the port
EXPOSE $PORT

# Command to run the API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "${PORT}"]