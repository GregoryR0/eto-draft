FROM python

ENV VERS=1.0

RUN apt-get update && \
	apt-get install -y vim

# What do we really need here - should prune this Greg
RUN \
	pip install --no-cache matplotlib && \
	pip install --no-cache matplotlib_venn && \
	pip install --no-cache pandas && \
	pip install --no-cache seaborn && \
	pip install --no-cache plotly && \
	pip install --no-cache plotly.express && \
	pip install --no-cache rasterio && \
	pip install --no-cache awscli && \
	pip install --no-cache s3fs && \
	pip install --no-cache ffmpeg && \
	pip install --no-cache pika && \
	pip install --no-cache dash && \
	pip install --no-cache boto3 && \
	pip install --no-cache fiona


ENV GREG_VERS=1.1
RUN mkdir -p /home/eto 

COPY etoLib /home/eto/etoLib
COPY api_eto /home/eto/api_eto

RUN (cd /home/eto/etoLib; make)

# Certificate Hell Fix!
RUN apt-get install ca-certificates && mkdir -p /etc/pki/tls/certs && \
	cp /etc/ssl/certs/ca-certificates.crt /etc/pki/tls/certs/ca-bundle.crt

WORKDIR /home/eto/api_eto
