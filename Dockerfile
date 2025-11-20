FROM prefecthq/prefect:3-python3.12

# Install system dependencies first
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libpq5 \
    git && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /opt/prefect

# Install dependencies directly (no uv needed for build, just runtime)
RUN pip install --no-cache-dir \
    "prefect>=3.4.24" \
    pandas \
    requests \
    python-jobspy \
    "prefect-dbt>=0.7.0" \
    sqlalchemy \
    anthropic \
    beautifulsoup4 \
    lxml \
    "psycopg2-binary==2.9.9" \
    "dbt-postgres>=1.8.1"

# Verify critical dependencies are installed
RUN python -c "import psycopg2; import prefect; import pandas; print('Dependencies OK')"

# DO NOT COPY flow code - Prefect will pull it from GitHub
