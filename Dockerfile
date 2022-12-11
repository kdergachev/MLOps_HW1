FROM python:3.9

RUN pip install poetry

RUN curl -sSL https://install.python-poetry.org | python3 -

COPY poetry.lock pyproject.toml .

RUN poetry config virtualenvs.create false \
  && poetry install --no-dev --no-interaction --no-ansi

COPY models models

COPY app.py trainable_models.py .

#CMD ["flask", "--app", "main.py", "run", "--host=0.0.0.0"]

EXPOSE 5000

CMD ["flask", "run", "-h", "0.0.0.0"]