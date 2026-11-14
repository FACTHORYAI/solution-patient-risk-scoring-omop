FROM python:3.12-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:0.13.0 /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
COPY data/sample ./data/sample
RUN uv sync --frozen --no-dev
RUN useradd --create-home runner && chown -R runner /app
USER runner
ENTRYPOINT ["uv", "run", "--no-dev", "python", "-m", "patient_risk_scoring_omop"]
