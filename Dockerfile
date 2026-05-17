# Multi-purpose image: same container can run the trader CLI or the Streamlit
# dashboard. Fly.io picks which one via the [processes] block in fly.toml.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TZ=UTC

WORKDIR /app

# System deps:
#   - tzdata so timezone math is correct (the trader aligns on UTC bars).
#   - procps for `ps`, used by app.py's orphan scanner.
#   - lsof for app.py's log-finder.
RUN apt-get update \
    && apt-get install -y --no-install-recommends tzdata procps lsof \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Streamlit listens here when the `app` process is used.
EXPOSE 8501

# Default to the trader. Override in fly.toml [processes] for the dashboard.
CMD ["python", "-u", "trade.py", "--help"]
