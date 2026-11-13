FROM kestra/kestra:latest-slim

# Switch to root to install system dependencies & python packages
USER root

WORKDIR /app

# Copy project files
COPY src   /app/src
COPY flows /app/flows
COPY examples /app/examples
COPY pyproject.toml  /app/
COPY requirements.txt /app/
COPY README.md       /app/

# Install dependencies using standard pip (cleaner than setting up uv in a Java container)
RUN apt-get update && apt-get install -y python3-pip && \
    pip install --no-cache-dir --break-system-packages -r /app/requirements.txt

# Drop back to the secure non-root Kestra user
USER kestra

ENV PYTHONPATH=/app
ENV APP_DIR=/app

EXPOSE 8080