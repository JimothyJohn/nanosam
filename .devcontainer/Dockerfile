FROM nvcr.io/nvidia/pytorch:23.01-py3

# Copy requirements.txt and install
COPY requirements.txt /tmp/
# upgrade pillow to fix "UnidentifiedImageError"
RUN pip install --no-cache-dir -U pillow pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt
