FROM python:3.10-slim

WORKDIR /app
ADD . /app
ARG MODEL_NAME="whisper-finetune-v2"

RUN mkdir "downloads"
RUN mkdir ${MODEL_NAME}
RUN pip install -U "huggingface_hub[cli]"
RUN huggingface-cli download "GuideU/${MODEL_NAME}" --local-dir ${MODEL_NAME}

# Install dependencies
RUN apt-get update -q \
    && apt-get install --no-install-recommends -qy python3-dev g++ gcc
RUN apt-get install -y ffmpeg
RUN pip install -r requirements.txt --progress-bar off

# Expose port
ENV PYTHONIOENCODING=utf-8
ENV PORT=8501

# Run the application:
CMD ["streamlit","run","app.py", "--server.maxUploadSize=100"]