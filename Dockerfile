FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install Poetry
RUN apt-get update && apt-get install -y curl && \
    curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry

# Copy files
RUN pip install poetry
COPY pyproject.toml poetry.lock* ./
# RUN poetry config virtualenvs.create false && poetry install --no-interaction --no-ansi
RUN poetry install --no-root
COPY . .

# Default run command
CMD ["poetry", "run", "python", "model/train.py"]
