#!/usr/bin/env bash
python3 -m unittest discover -s tests -p '*test*.py'
python3 -m unittest discover -s tests -p '*Test*.py'
