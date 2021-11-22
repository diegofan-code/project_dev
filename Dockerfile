FROM python:3.8
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD python3 validate_ml.py