#!/bin/bash
echo "Python version:"
python --version
echo "Upgrading pip..."
pip install --upgrade pip
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt
echo "Setup completed successfully!"