#!/bin/bash

# Create pip directory in AppData if it doesn't exist
mkdir -p "%APPDATA%\pip"

# Create pip.ini file with PyTorch CPU index
echo "[global]
extra-index-url = https://download.pytorch.org/whl/cpu" > "%APPDATA%\pip\pip.ini"

export DISTUTILS_USE_SDK=1