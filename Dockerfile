FROM prefecthq/prefect:3-python3.12
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libpq5 \
    git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/prefect

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
    "dbt-postgres>=1.8.1" \
    rapidfuzz

# Patch jobspy LinkedIn bug: job_level can be None, .lower() crashes
RUN sed -i 's/job_details.get("job_level", "").lower()/(job_details.get("job_level") or "").lower()/' \
    /usr/local/lib/python3.12/site-packages/jobspy/linkedin/__init__.py

COPY . /opt/prefect/

RUN pip install --no-cache-dir ./vendor/llm-queue-client
