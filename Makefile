# drift-vs-harm
#
# `make verify` is the one command the README promises a reviewer: it runs with
# no network from a fresh clone and must stay green. It installs no evidence
# extras and imports no optional dependency.

PY ?= python
PKG = src/driftharm

.PHONY: help install lint test receipts verify experiments evidence clean

help:
	@echo "install     editable install with the test extra"
	@echo "lint        ruff check (line length 100)"
	@echo "test        pytest with junitxml + coverage json, 100% line floor"
	@echo "experiments re-run exp01..exp05, writing docs/experiments/*.json"
	@echo "receipts    re-measure every published number and check it against the docs"
	@echo "verify      lint + test + receipts  (the clone-run-verify contract, no network)"
	@echo "evidence    dev-time only: render diagram SVG, screenshots, replay GIF"

install:
	$(PY) -m pip install -e ".[test]"

lint:
	ruff check src tests experiments tools

test:
	$(PY) -m pytest tests \
		--junitxml=reports/junit.xml \
		--cov=driftharm --cov-report=json:reports/coverage.json \
		--cov-report=term-missing
	$(PY) tools/check_coverage.py reports/coverage.json 100

experiments:
	$(PY) experiments/exp01_silent_failure.py
	$(PY) experiments/exp02_false_alarm.py
	$(PY) experiments/exp03_magnitude_decoupling.py
	$(PY) experiments/exp04_binning_sensitivity.py
	$(PY) experiments/exp05_label_delay_window.py

receipts:
	$(PY) tools/collect_metrics.py --skip-tests
	$(PY) tools/check_numbers.py

# The verify contract. Tests first (they produce the coverage and junit reports
# receipts reads), then experiments re-measured, then the numbers checked.
verify: lint test experiments receipts

# Evidence is not part of verify. It needs a browser and node and is slow; CI
# runs pure Python. See docs/adr/ADR-002 and references in the tools.
evidence:
	$(PY) tools/render_diagrams.py
	$(PY) tools/capture_screenshots.py
	$(PY) tools/record_demo.py

clean:
	rm -rf reports .pytest_cache .ruff_cache coverage.json junit.xml
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
