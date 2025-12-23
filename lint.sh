isort src
black src
ruff check src --fix
mypy src