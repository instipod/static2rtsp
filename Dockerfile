FROM python:3.14

RUN apt-get update && apt-get install ffmpeg -y && rm -rf /var/cache/apt/archives /var/lib/apt/lists/*
COPY requirements.txt /tmp/requirements.txt
RUN cd /tmp && pip3 install -r requirements.txt && rm /tmp/requirements.txt

COPY main.py /usr/local/bin/main.py
RUN chmod a+x /usr/local/bin/main.py

ENV PYTHONUNBUFFERED=1
CMD ["/usr/local/bin/main.py"]
