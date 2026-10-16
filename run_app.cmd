@echo off
rem מפעיל את האפליקציה במחשב המקומי. אפשר ללחוץ על הקובץ פעמיים.
cd /d "%~dp0"
python -X utf8 -m streamlit run app.py --server.port 8501
