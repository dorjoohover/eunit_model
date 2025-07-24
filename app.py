from flask import Flask
from config import HOST, PORT, API_KEY
from routes.health import health_bp
from routes.predict import predict_bp

app = Flask(__name__)
app.config["API_KEY"] = API_KEY

# register blueprints
app.register_blueprint(health_bp)
app.register_blueprint(predict_bp)

if __name__ == "__main__":
    # for dev/testing only
    app.run(host=HOST, port=PORT)
