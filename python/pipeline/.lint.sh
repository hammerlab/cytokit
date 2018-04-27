#!/bin/bash
set -o errexit

find . -name '*.py' \
  | xargs pylint \
  --errors-only \
  --disable=unsubscriptable-object,not-an-iterable,no-member

echo 'Passes pylint check'