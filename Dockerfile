FROM python:3.10-slim

WORKDIR /app
ADD . /app
ARG MODEL_NAME="whisper_wak"

RUN mkdir "downloads"
RUN mkdir ${MODEL_NAME}
RUN pip install -U "huggingface_hub[cli]"
RUN huggingface-cli download "GuideU/${MODEL_NAME}" --local-dir ${MODEL_NAME}


ARG MODEL_NAME_2="whisper-mobi-240418"
RUN mkdir ${MODEL_NAME_2}
RUN huggingface-cli download "GuideU/${MODEL_NAME_2}" --local-dir ${MODEL_NAME_2}

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