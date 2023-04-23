#!/bin/bash

if [ -n "$(git status --porcelain)" ]; then
    echo "There are uncommitted changes in working tree after execution of the build"
    git --no-pager diff
    git status
    echo "Please commit all changes..."
    exit 1
else
    echo "Git working tree is clean"
fi