FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml .
COPY confab/ confab/
COPY confab.py .

RUN pip install --no-cache-dir . starlette uvicorn

ENV PORT=8080
EXPOSE 8080

CMD ["python", "-m", "confab", "proxy", "--port", "8080"]
