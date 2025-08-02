# Makefile script
# ==============
# 
# Author: Akshay Mestry <xa@mes3.dev>
# Created on: Wednesday, July 30 2025
# Last updated on: Saturday, August 02 2025
# 
# This Makefile provides a comprehensive set of commands (targets) for managing
# the development workflow, including cleaning, testing, linting, formatting,
# type checking, and packaging.

.DEFAULT_GOAL := help
.PHONY: help clean test coverage lint format typecheck install all

BUILD_DIR := build
DIST_DIR := dist
DOCS_DIR := docs
PIP := pip
PYTHON := python3
PROJECT_NAME := lauren
TEST_DIR := tests

COLOUR_BLUE := \033[34m
COLOUR_BOLD := \033[1m
COLOUR_CYAN := \033[36m
COLOUR_DIM := \033[2m
COLOUR_GREEN := \033[32m
COLOUR_MAGENTA := \033[35m
COLOUR_RED := \033[31m
COLOUR_RESET := \033[0m
COLOUR_WHITE := \033[37m
COLOUR_YELLOW := \033[33m

TARGET_MAX_CHAR_NUM := 15
TERMINAL_WIDTH := $(shell tput cols)

# Function to show animated ellipsis during operations
# Usage: $(call animate, message, command, sleep)
define animate
	@printf "$(1)"; \
	(FORCE_COLOR=1 TERM=xterm-256color PY_COLORS=1 $(2)) > /tmp/make_output 2>&1 & \
	pid=$$!; \
	while kill -0 $$pid 2>/dev/null; do \
		for dots in "." ".." "..."; do \
			printf "\r$(1)$$dots   "; \
			sleep $(3); \
			if ! kill -0 $$pid 2>/dev/null; then break 2; fi; \
		done; \
	done; \
	wait $$pid; exit_code=$$?; \
	printf "\r$(1)... "; \
	if [ $$exit_code -eq 0 ]; then \
		printf "$(COLOUR_GREEN)done!$(COLOUR_RESET)\n"; \
	else \
		printf "$(COLOUR_RED)failed!$(COLOUR_RESET)\n"; \
	fi; \
	cat /tmp/make_output 2>/dev/null || true; \
	rm -f /tmp/make_output; \
	exit $$exit_code
endef

help: ## Display this help and available commands
	@echo "Usage:"
	@echo "  $(COLOUR_GREEN)make$(COLOUR_RESET) $(COLOUR_RED)<target>$(COLOUR_RESET)"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | \
		sort | \
		awk 'BEGIN {FS = ":.*## *"} {printf "  $(COLOUR_YELLOW)%-$(TARGET_MAX_CHAR_NUM)s$(COLOUR_RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(COLOUR_DIM)For executing a particular target, run 'make <target>'.$(COLOUR_RESET)"
	@echo "$(COLOUR_DIM)Read complete documentation at: https://github.com/xames3/lauren.$(COLOUR_RESET)"

clean: ## Remove build artifacts, cache files, and temporary directories
	$(call animate,Cleaning build artifacts and cache files, \
		rm -rf $(BUILD_DIR)/ $(DIST_DIR)/ *.egg-info/ .coverage htmlcov/ .pytest_cache/ .mypy_cache/ .ruff_cache/ 2>/dev/null && \
		find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true && \
		find . -type f -name "*.pyc" -delete 2>/dev/null || true && \
		find . -type f -name "*.pyo" -delete 2>/dev/null || true, \
		0.3 \
	)

install: ## Install package in editable (development) mode
	$(call animate,Installing $(COLOUR_BOLD)$(PROJECT_NAME)$(COLOUR_RESET) in editable mode, \
		$(PIP) install --editable .[test] --quiet --force-reinstall && \
		$(PIP) install --editable .[dev] --quiet --force-reinstall, \
		0.3 \
	)

test: ## Execute test suite with coverage reporting
	$(call animate,Running $(COLOUR_CYAN)pytest$(COLOUR_RESET) tests suite, \
		COLUMNS=$(TERMINAL_WIDTH) pytest, \
		0.3 \
	)

tox: ## Execute tests across multiple Python environments using tox
	$(call animate,Running tests across different virtual environments (with $(COLOUR_MAGENTA)coverage$(COLOUR_RESET) enabled), \
		tox -p -qq, \
		0.3 \
	)
coverage: ## Generate comprehensive test coverage reports in multiple formats
	$(call animate,Checking code $(COLOUR_MAGENTA)coverage$(COLOUR_RESET), \
		COLUMNS=$(TERMINAL_WIDTH) tox -qq -e coverage, \
		0.3 \
	)
format: ## Apply automatic code formatting using ruff
	$(call animate,Formatting code with $(COLOUR_RED)ruff$(COLOUR_RESET), \
		ruff format $(PROJECT_NAME) $(TEST_DIR), \
		0.3 \
	)

lint: ## Perform comprehensive linting checks and report issues
	$(call animate,Running linting checks with $(COLOUR_RED)ruff$(COLOUR_RESET), \
		ruff check $(PROJECT_NAME) $(TEST_DIR) --output-format=full --quiet, \
		0.3 \
	)

typecheck: ## Perform static type checking with mypy
	$(call animate,Performing static type analysis using with $(COLOUR_GREEN)mypy$(COLOUR_RESET), \
		mypy $(PROJECT_NAME) --config-file pyproject.toml --color-output --show-error-codes, \
		0.3 \
	)

benchmark: ## Execute performance benchmarks and profiling tests
	$(call animate,Running performance benchmarks, \
		COLUMNS=$(TERMINAL_WIDTH) pytest -k "performance" --benchmark-only --color=yes, \
		0.3 \
	)

check: ## Perform comprehensive code quality assessment
	@$(MAKE) lint --no-print-directory
	@$(MAKE) format --no-print-directory
	@$(MAKE) typecheck --no-print-directory
	@$(MAKE) test --no-print-directory
	$(call animate,Finished all code quality checks,sleep 1,0.3)

release-check: ## Verify package readiness for distribution release
	@$(MAKE) check --no-print-directory
	$(call animate,Verifying release readiness, \
		$(PYTHON) -m build --wheel --sdist &&\
		$(PYTHON) -m twine check $(DIST_DIR)/*, \
		0.3 \
	)

all: ## Execute complete development workflow: clean, install, and check
	@$(MAKE) clean --no-print-directory
	@$(MAKE) install --no-print-directory
	@$(MAKE) check --no-print-directory
	$(call animate,Complete development workflow,sleep 1,0.3)
