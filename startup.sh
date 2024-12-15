gunicorn --workers 3 --bind 0.0.0.0:8000 app:app --log-level debug
