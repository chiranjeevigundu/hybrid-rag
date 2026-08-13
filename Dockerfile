# The retrieval service, for anywhere that runs containers.
#
# The one non-obvious thing here is that the embedding model is downloaded at *build*
# time, not at startup. fastembed otherwise fetches ~130 MB from HuggingFace the first
# time TextEmbedding is constructed, which in a scheduler like ECS or Kubernetes means
# every task start pays it — and a task that dies mid-download is restarted into the
# same download. Baking it in makes the image bigger and the container's startup
# deterministic and offline, which is the right trade for a service that is scaled by
# replacing tasks.

FROM python:3.12-slim AS build

WORKDIR /src
COPY pyproject.toml README.md ./
COPY ragkit ./ragkit

# `serve` brings FastAPI and uvicorn; `embed` brings fastembed and onnxruntime. The
# base dependency set is just psycopg, which is deliberate: the library has to stay
# importable for the eval harness and batch indexing jobs without pulling a web server in.
RUN pip install --no-cache-dir --prefix=/install ".[serve,embed]"


FROM python:3.12-slim

# Model cache lives outside the home directory so it survives a user change and can be
# mounted read-only. RAG_CACHE_DIR is what ragkit.config reads.
ENV RAG_CACHE_DIR=/opt/models \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY --from=build /install /usr/local
WORKDIR /app
COPY pyproject.toml README.md ./
COPY ragkit ./ragkit
COPY migrations ./migrations

# Resolve the model id from the code rather than repeating it here. If DEFAULT_MODEL
# ever changes, the image follows it; a hardcoded string would silently bake the wrong
# weights and the service would download the right ones at startup, which is exactly
# the failure this stage exists to prevent.
RUN python -c "\
from fastembed import TextEmbedding; \
from ragkit.config import DEFAULT_MODEL; \
print('baking', DEFAULT_MODEL); \
TextEmbedding(model_name=DEFAULT_MODEL, cache_dir='/opt/models')" \
    && python -c "import ragkit.service"

# Non-root. The service reads a corpus and a database; it writes nothing to disk.
RUN useradd --system --create-home --uid 10001 ragkit \
    && chown -R ragkit:ragkit /opt/models
USER ragkit

EXPOSE 8080

# The service builds the ONNX session and the connection pool in its lifespan hook, so
# /health answering means it is genuinely ready to serve a query, not merely bound to a
# port. That is what makes it usable as a load balancer health check.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=4).status==200 else 1)"

CMD ["uvicorn", "ragkit.service:app", "--host", "0.0.0.0", "--port", "8080"]
