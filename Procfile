web: PYTHONPATH=. gunicorn inference.app_deploy:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120 --preload
