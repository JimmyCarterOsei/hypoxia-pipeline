# Convenience targets. Nothing here is required - CI calls the same commands.
IMAGE_PY  ?= ghcr.io/OWNER/hypoxiapipe
IMAGE_R   ?= ghcr.io/OWNER/hypoxiapipe-r
TAG       ?= dev

.PHONY: help test lint typecheck check images image-py image-r smoke clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

test:  ## Run the test suite (offline)
	pytest

lint:  ## Lint and format check
	ruff check src tests && ruff format --check src tests

typecheck:  ## Strict type check
	mypy

check: lint typecheck test  ## Everything CI runs

image-py:  ## Build the Python stage image
	docker build -f docker/Dockerfile.python -t $(IMAGE_PY):$(TAG) .

image-r:  ## Build the R survival stage image
	docker build -f docker/Dockerfile.r -t $(IMAGE_R):$(TAG) .

images: image-py image-r  ## Build both images

smoke: images  ## Run the containerised R stage from the Python stage
	docker run --rm -i $(IMAGE_R):$(TAG) < tests/fixtures/r_request.json | head -c 400; echo
	docker run --rm $(IMAGE_PY):$(TAG) sig list

clean:  ## Remove caches
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
