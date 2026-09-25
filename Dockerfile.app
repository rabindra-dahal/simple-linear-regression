# Use an official lightweight Python runtime
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /workspace

# Copy dependencies first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Streamlit app code
COPY app.py .

# Expose Streamlit default port
EXPOSE 8501

# Start Streamlit hosting on all network interfaces
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
