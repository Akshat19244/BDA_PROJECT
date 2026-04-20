FROM python:3.10-slim

# Install Java (stable package name)

RUN apt-get update && apt-get install -y default-jdk

# Set Java path automatically

ENV JAVA_HOME=/usr/lib/jvm/default-java

# Install PySpark

RUN pip install pyspark

WORKDIR /app

COPY . /app

CMD ["python", "app.py"]
