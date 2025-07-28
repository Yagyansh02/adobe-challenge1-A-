@echo off
REM Set UTF-8 encoding for Python
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

REM Run the test script
python test_all_pdfs.py %*

pause
