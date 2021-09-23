FROM python:3.8-slim
COPY . .
RUN apt-get update && apt-get install -y \
    python3-pip && pip3 install -r requirements.txt
CMD ["python3", "source/personality_predictor.py"]