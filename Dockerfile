FROM python:3.12.1-slim-bullseye

RUN pip install --upgrade pip
RUN pip install pipenv

WORKDIR /ia_backend

COPY . .

RUN pipenv install --system --deploy --ignore-pipfile

# Set python path in bashrc
ENV PYTHONPATH "${PYTHONPATH}:/ia_backend"

# Run unnittests
# if the tests fail, the build will fail
RUN pipenv run test

# expose port 8000
EXPOSE 8000

# Run the server
CMD ["pipenv", "run", "start_prod"]

