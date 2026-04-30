FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    OUTPUT_ROOT=/mnt/pai/outputs \
    MODEL_DIR=/mnt/pai/models/Llama-3.1-8B \
    HF_HOME=/mnt/pai/cache/hf \
    TRANSFORMERS_CACHE=/mnt/pai/cache/hf \
    EAS_TASK_MODE=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml README.md /app/
COPY src /app/src
COPY config /app/config
COPY data /app/data
COPY run_ai_re_judge.py /app/
COPY run_inference.py /app/
COPY run_llamascope_distribution_sanity.py /app/
COPY run_llamascope_variant_audit.py /app/
COPY run_sae_checkpoint_parity.py /app/
COPY run_sae_evaluation.py /app/
COPY run_sae_forward_chain_audit.py /app/

RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124 && \
    python -m pip install --no-cache-dir -r requirements.txt && \
    python -m pip install --no-cache-dir -e .

EXPOSE 8000

CMD ["uvicorn", "nlp_re_base.pai.service_api:app", "--host", "0.0.0.0", "--port", "8000"]
