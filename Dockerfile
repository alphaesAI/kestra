#   Kestra‑based Docker image for the FileNest Dev pipeline

# Use the official Kestra base image (includes the Kestra engine)
FROM kestra/kestra:latest

# ------------ Set work directory ------------
WORKDIR /app

# ------------ Copy source code ------------
# Copy the entire Kestra project (src, flows, examples, prompts, etc.)
COPY src   /app/src
COPY flows /app/flows
COPY examples /app/examples
COPY prompts /app/prompts

# ------------ Copy dependency files ------------
COPY pyproject.toml  /app/
COPY requirements.txt /app/
COPY README.md       /app/

# ------------ Install UV (fast Python installer) ------------
# UV is a tiny, fast installer; we use it to install the deps inside the base image
RUN apt-get update && apt-get install -y curl && \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    /root/.cargo/bin/uv pip install --system -r /app/requirements.txt

# ------------ Make the project importable ------------
ENV PYTHONPATH=/app
# Kestra injects a runtime env var `APP_DIR` that points to the container root.
# We expose it here for local debugging; the default is /app.
ENV APP_DIR=/app

# ------------ Expose Kestra HTTP port ------------
EXPOSE 8080

# ------------ Default command (run Kestra) ------------
CMD ["java", "-jar", "/kestra/kestra.jar"]