#!/bin/bash

FILES="experiments/*"
for f in $FILES
do
  case "$f" in
  "__init__.py")
    ;;
  *)
    echo "Processing $f..."
    python3 "$f"
    ;;
  esac
done