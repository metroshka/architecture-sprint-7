FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HOME=/tmp/huggingface \
    RAG_CONTAINER=1

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /app/requirements.txt
# Версия torch берётся из общего файла; устанавливается сборка без CUDA.
RUN python -c "from pathlib import Path; Path('/tmp/torch.txt').write_text(''.join(line+'\n' for line in Path('requirements.txt').read_text().splitlines() if line.startswith('torch==')))" \
    && python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu -r /tmp/torch.txt \
    && python -m pip install --no-cache-dir -r requirements.txt \
    && python -m pip check

RUN useradd --create-home --uid 10001 app \
    && mkdir -p /app/reports /app/data/injection_lab /app/models/multilingual-e5-small \
    && chown -R app:app /app
COPY --chown=app:app scripts/ /app/scripts/
COPY --chown=app:app config/ /app/config/
COPY --chown=app:app prompts/ /app/prompts/
COPY --chown=app:app knowledge_base/ /app/knowledge_base/
COPY --chown=app:app index/ /app/index/
COPY --chown=app:app data/rag_cases.json data/retrieval_cases.json /app/data/
COPY --chown=app:app data/security/ /app/data/security/
COPY --chown=app:app Dockerfile docker-compose.yml /app/
USER app
RUN python scripts/check_index_environment.py
CMD ["python", "scripts/rag_bot.py"]
