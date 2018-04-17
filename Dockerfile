FROM jjanzic/docker-python3-opencv:opencv-3.4.1

RUN python -mpip install -U pip && \
    python -mpip install -U matplotlib

RUN mkdir -p /var/lib/py-scripts && \
    mkdir -p /var/lib/py-images && \
    mkdir -p /var/lib/py-volumes

VOLUME /var/lib/py-scripts
VOLUME /var/lib/py-images
VOLUME /var/lib/py-volumes

CMD ["ping", "localhost"]
