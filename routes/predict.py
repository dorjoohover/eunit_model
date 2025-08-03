from utils.transformers import fuel_values
from datetime import datetime
from zeep.helpers import serialize_object
from config import ACCESS_TOKEN, KEY_PATH
from flask import Blueprint, request, jsonify, abort
from services.prediction import predict
from utils.security import require_api_key
from services.xyp import Service
from services.normalize import normalize_mark, normalize_brand
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
def predict_post():
    try:
        body = request.get_json()
        if not body:
            abort(400, description="Invalid or missing JSON body")
        print(body.get('num'))
        vehicle = getVehicle(body.get('num'))
        milleage, drive, gearbox = body.get(
            'milleage'), body.get('drive'), body.get('gearbox')
        print(vehicle)
        vin = vehicle.get('cabinNumber')
        brand, mark,  buildYear = vehicle.get('markName'), vehicle.get(
            'modelName'),  vehicle.get('buildYear')
        importedDate = datetime.strptime(
            str(vehicle.get('importDate')), '%Y-%m-%d %H:%M:%S%z').year
        khurd = 'Буруу' if vehicle.get('wheelPosition') == 'Баруун' else 'Зөв'
        color = vehicle.get('colorName')
        capacity = round(float(str(vehicle.get('capacity'))) / 1000, 1)
        engine = fuel_values(vehicle.get('fueltype'))
        normalized = normalize_mark(brand, mark, vin)
        brand = normalize_brand(brand)
        features = {
            'brand': brand,
            'mark': normalized,
            'Engine_capacity': capacity,
            'Year_of_manufacture': buildYear,
            'Year_of_entry': importedDate,
            'Gearbox': gearbox,
            'Hurd': khurd,
            'Type': None,
            'Color': color,
            'Engine': engine,
            'Interior_color': None,
            'Drive': drive,
            'Mileage': milleage,
            'Conditions': None
        }
        print(features)
        # return features
        result = predict(features)
        return jsonify({"prediction": result, "vehicle": vehicle, "features": features}), 200

    except Exception as e:
        print(e)
        abort(500, description=str(e))


@predict_bp.route("/vehicle", methods=["POST"])
def vehicle():
    try:
        body = request.get_json()
        if not body:
            abort(400, description="Invalid or missing JSON body")
        vehicle = getVehicle(body.get('num'))

        return jsonify({"vehicle": vehicle}), 200

    except Exception as e:
        print(e)
        abort(500, description=str(e))
        
        
@predict_bp.route("/property", methods=["POST"])
def pro():
    try:
        body = request.get_json()
        if not body:
            abort(400, description="Invalid or missing JSON body")
        p = getPropertyInfo(body.get('property'))

        return jsonify({"property": p}), 200

    except Exception as e:
        print(e)
        abort(500, description=str(e))


def getVehicle(arg: str = ''):
    body = request.get_json()
    if not body:
        abort(400, description="Invalid or missing JSON body")

    arg = body.get('num')
    if not arg:
        abort(400, description="Missing `num` field")

    try:
        params = {
            "auth": None,
            "cabinNumber": None,
            "certificatNumber": None,
            "regnum": None,
        }
        if len(arg) <= 7:
            params.update({'plateNumber': arg})
        else:
            params.update({'certificateNumber': arg})

        # ACCESS_TOKEN, KEY_PATH - үнэн эсэхийг шалгах
        if not ACCESS_TOKEN or not KEY_PATH:
            return jsonify({"error": "ACCESS_TOKEN or KEY_PATH is missing"}), 500

        citizen = Service(
            'https://xyp.gov.mn/transport-1.3.0/ws?WSDL',
            ACCESS_TOKEN,
            KEY_PATH
        )
        print(citizen)
        res = citizen.dump('WS100401_getVehicleInfo', params).response
        res_dict = serialize_object(res)
        print(res_dict)
        return res_dict

    except Exception as e:
        print("getVehicle error:", str(e))
        return jsonify({"error": str(e)}), 500


def getPropertyInfo(arg: str = ''):
    body = request.get_json()
    if not body:
        abort(400, description="Invalid or missing JSON body")

    arg = body.get('property')
    if not arg:
        abort(400, description="Missing `property` field")

    try:
        params = {
            "auth": None,
            "regnum": None,
        }
        params.update({"propertyNumber": arg})

        print(params)
        # ACCESS_TOKEN, KEY_PATH - үнэн эсэхийг шалгах
        if not ACCESS_TOKEN or not KEY_PATH:
            return jsonify({"error": "ACCESS_TOKEN or KEY_PATH is missing"}), 500
        print(params)
        citizen = Service(
            'https://xyp.gov.mn/property-1.3.0/ws?WSDL',
            ACCESS_TOKEN,
            KEY_PATH
        )
        print(citizen)
        res = citizen.dump('WS100201_getPropertyInfo', params).response
        res_dict = serialize_object(res)
        print(res_dict)
        return res_dict

    except Exception as e:
        print("getVehicle error:", str(e))
        return jsonify({"error": str(e)}), 500
