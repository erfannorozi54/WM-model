@echo off
REM Activation script for Working Memory Model project

call /home/erfan/Projects/WM-model/venv\Scripts\activate.bat
set PYTHONPATH=/home/erfan/Projects/WM-model\src;%PYTHONPATH%
echo Working Memory Model environment activated!
echo Project directory: /home/erfan/Projects/WM-model
echo Python path includes: /home/erfan/Projects/WM-model\src
