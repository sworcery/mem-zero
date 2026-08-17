.PHONY: lock test lint

## Re-resolve the Docker image's Python dependencies. pyproject.toml ranges stay
## floating; requirements.txt is the pinned set the image installs (py3.12).
lock:
	uv pip compile --universal --python-version 3.12 --upgrade -o requirements.txt pyproject.toml

test:
	PYTHONPATH=src python3 -m pytest -q

lint:
	python3 -m ruff check src/ tests/
	python3 -m mypy src/mem_zero
