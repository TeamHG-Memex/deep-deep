#!/usr/bin/env bash

set -ev

# Running tests...
py.test --cov=deepdeep --cov-report=html --cov-report=term \
        --doctest-modules deepdeep

# Running type checks...
mypy deepdeep --warn-unused-ignores
