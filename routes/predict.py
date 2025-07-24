from flask import Blueprint, request, jsonify, abort
from services.prediction import predict
from utils.security import require_api_key
from services.xyp import getVehicle

predict_bp = Blueprint("predict", __name__)

# Example GET route with hardcoded features
@predict_bp.route("/predict", methods=["GET"])
# @require_api_key()
def predict_get():
    features = {
        'brand': 'toyota',
        'mark': 'prius-41',
        'Engine_capacity': 1.8,
        'Year_of_manufacture': 2012,
        'Year_of_entry': 2023,
        'Gearbox': 'Автомат',
        'Hurd': 'Буруу',
        'Type': 'Гэр бүлийн',
        'Color': 'Саарал',
        'Engine': 'Бензин',
        'Interior_color': 'Хар',
        'Drive': 'Урдаа FWD',
        'Mileage': 180000,
        'Conditions': 'Дугаар авсан'
    }
    try:
        result = predict(features)
        return jsonify({"prediction": result}), 200
    except Exception as e:
        abort(500, description=str(e))

# POST route that takes input from client
@predict_bp.route("/predict/car", methods=["POST"])
# @require_api_key()
async def predict_post():
    try:
        body = request.get_json()
        if not body:
            abort(400, description="Invalid or missing JSON body")
        print(body.get('num'))
        vehicle = await getVehicle(body.get('num'))  # optional, if used

        features = {
            'brand': body.get('brand'),
            'mark': body.get('mark'),
            'Engine_capacity': body.get('Engine_capacity'),
            'Year_of_manufacture': body.get('Year_of_manufacture'),
            'Year_of_entry': body.get('Year_of_entry'),
            'Gearbox': body.get('Gearbox'),
            'Hurd': body.get('Hurd'),
            'Type': body.get('Type'),
            'Color': body.get('Color'),
            'Engine': body.get('Engine'),
            'Interior_color': body.get('Interior_color'),
            'Drive': body.get('Drive'),
            'Mileage': body.get('Mileage'),
            'Conditions': body.get('Conditions')
        }

        result = predict(features)
        return jsonify({"prediction": result}), 200

    except Exception as e:
        abort(500, description=str(e))
