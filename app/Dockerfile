FROM python:3.7-slim-buster

RUN apt-get update &&\
    apt-get install -y python3-dev build-essential

# make app directory
RUN mkdir -p /usr/src/toxicityclassifier

# change directory
WORKDIR /usr/src/toxicityclassifier

# copy and install requirements
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# copy all files over
COPY . .

# streamlit specific commands for config
CMD python app.py