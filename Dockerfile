FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORTAL_DATA_DIR=/data PORTAL_APP_ROOT=/app
WORKDIR /app

COPY pyproject.toml README.md ./
COPY aitesteval ./aitesteval
COPY portal ./portal
COPY scenarios ./scenarios
COPY datasets ./datasets
COPY examples ./examples
RUN pip install --no-cache-dir ".[portal]" \
 && useradd --create-home --uid 10001 app && mkdir -p /data && chown app /data
USER app

VOLUME ["/data"]
# The port is $PORT when the host sets it (Render, Heroku-style platforms), else 8000.
# `python -m portal` reads $PORT itself, so no shell is needed to expand it.
EXPOSE 8000
HEALTHCHECK CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/api/v1/health' % (os.environ.get('PORT') or 8000))"
CMD ["python", "-m", "portal", "--host", "0.0.0.0"]
