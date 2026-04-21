# Hugging Face Spaces — Docker SDK — runs the VeriVision Streamlit app.
# HF Spaces expects the container to listen on port 7860.

FROM python:3.11-slim

# System deps — libgl for pillow/matplotlib on some HF runners
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces runs as a non-root user named "user" with UID 1000
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

# Copy requirements first for better layer caching
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY --chown=user . .

USER user
ENV PATH="/home/user/.local/bin:${PATH}" \
    STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 7860

CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
