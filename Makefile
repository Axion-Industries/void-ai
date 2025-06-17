.PHONY: install test lint clean

install:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

test:
	pytest --cov=. --cov-report=term-missing

lint:
	black .
	isort .
	flake8 .
	pylint *.py

clean:
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} +
	find . -type d -name "*.egg" -exec rm -r {} +
	find . -type d -name ".pytest_cache" -exec rm -r {} +
	find . -type d -name ".mypy_cache" -exec rm -r {} +

run:
	python chat_api.py

train:
	python train_void.py
