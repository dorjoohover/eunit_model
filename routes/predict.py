from flask import Blueprint, request, jsonify, abort
from services.prediction import predict
from utils.security import require_api_key
from services.xyp import Service
predict_bp = Blueprint("predict", __name__)
from config import ACCESS_TOKEN, KEY_PATH
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
def predict_post():
    try:
        body = request.get_json()
        if not body:
            abort(400, description="Invalid or missing JSON body")
        print(body.get('num'))
        # vehicle =getVehicle(body.get('num'))  
        # print(vehicle)
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
@predict_bp.route("/predict/vehicle", methods=["POST"])

def getVehicle():
    body = request.get_json()
    if not body:
        abort(400, description="Invalid or missing JSON body")
    arg =  body.get('num')
    try:
        params = {
            "auth": None,
            "cabinNumber": None,
            "certificatNumber": None,
            "regnum": None,
        }
        # арг 7 оронтой бол plates, урт бол гэрчилгээ
        if len(arg) <= 7:
            params.update({'plateNumber': arg})
        else:
            params.update({'certificateNumber': arg})

        print("📤 Params:", params)

        citizen = Service(
            'https://xyp.gov.mn/transport-1.3.0/ws?WSDL',
            ACCESS_TOKEN,
            KEY_PATH
        )

        res = citizen.dump('WS100401_getVehicleInfo', params)
        print("📥 Response:", res)
        return jsonify({"data": res})

    except Exception as e:
        print(f"getVehicle error:", str(e))
        return None
