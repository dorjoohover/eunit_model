from flask import Flask
from config import HOST, PORT, API_KEY
from routes.health import health_bp
from routes.predict import predict_bp
from concurrent.futures import ThreadPoolExecutor

# Flask app initialization
app = Flask(__name__)
app.config["API_KEY"] = API_KEY

# ThreadPoolExecutor for background tasks
executor = ThreadPoolExecutor(max_workers=4)
app.config["EXECUTOR"] = executor

# Register blueprints
app.register_blueprint(health_bp)
app.register_blueprint(predict_bp)

# Optional: Root route
@app.route("/")
def home():
    return {"status": "✅ Flask app is running!"}

# Run only if executed directly
if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=True)
