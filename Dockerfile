FROM python:3.10-slim

WORKDIR /app
ADD . /app

# Install dependencies
RUN apt-get update -q \
    && apt-get install --no-install-recommends -qy python3-dev g++ gcc
RUN pip install -r requirements.txt --progress-bar off
RUN apt-get install -y ffmpeg

# Expose port 
ENV PORT=8501

# Run the application:
CMD ["streamlit","run","app.py"]