#!/usr/bin/env bash

echo "OSTYPE = ${OSTYPE}"
if [[ "$OSTYPE" == "darwin"* ]]; then
    #mac zsh
    echo "OS = MacOS"
    source ./venv/bin/activate
elif [[ "$OSTYPE" == "msys" ]]; then
    #windows mingw/git bash
    echo "OS = Windows"
    source ./venv/Scripts/activate
    which python
else
    echo "OS not found."
    exit
fi

pip freeze > requirements.txt