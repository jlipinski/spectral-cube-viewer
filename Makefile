# Convenience targets (Linux/macOS; on Windows run the commands after the tabs directly).
PYTHON ?= python3

.PHONY: install dev test test-all run catalog build clean

install:            ## install the application
	$(PYTHON) -m pip install .

dev:                ## editable install with all extras, for development
	$(PYTHON) -m pip install -e ".[online,catalog,test]"

test:               ## run the test suite (offline)
	QT_QPA_PLATFORM=offscreen $(PYTHON) -m pytest

test-all:           ## run the test suite including tests that need internet
	RUN_NETWORK_TESTS=1 QT_QPA_PLATFORM=offscreen $(PYTHON) -m pytest

run:                ## start the GUI
	$(PYTHON) -m spectral_cube_viewer

catalog:            ## re-download the offline line catalog (needs internet)
	$(PYTHON) tools/build_line_catalog.py

build:              ## build sdist and wheel into dist/
	$(PYTHON) -m pip install --upgrade build
	$(PYTHON) -m build

clean:
	rm -rf build dist src/*.egg-info .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
