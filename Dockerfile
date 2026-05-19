# Use the official Python 3.10 image as the base
FROM python:3.10-slim

# Set up a new user named "user" with user ID 1000
# Hugging Face Spaces requires running as a non-root user
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY --chown=user . .

# Expose port 7860, which is required by Hugging Face Spaces
EXPOSE 7860

# Set environment variables for the application
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# Run the FastAPI server on port 7860
CMD ["uvicorn", "web.app:app", "--host", "0.0.0.0", "--port", "7860"]
