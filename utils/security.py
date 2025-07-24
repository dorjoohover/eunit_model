from functools import wraps
from flask import request, abort, current_app

def require_api_key():
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            key = request.headers.get("X-API-KEY")
            if not key or key != current_app.config["API_KEY"]:
                abort(401, description="Invalid or missing API key")
            return f(*args, **kwargs)
        return wrapped
    return decorator
