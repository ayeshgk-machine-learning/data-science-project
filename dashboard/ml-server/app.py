# dependencies
from statistics import median
from flask import Flask, request, jsonify, make_response
from joblib import load
import numpy as np
from flask_cors import CORS
import pandas as pd


from flask_json_schema import JsonSchema, JsonValidationError
from flask import Response

app = Flask(__name__)
CORS(app)
schema = JsonSchema(app)


x_columns = {'account_length': {"min": 0, "max": 250, "median": 101.0},
             'intertiol_plan': False,
             'voice_mail_plan': False,
             'number_vm_messages': {"min": 0, "max": 51, "median": 0},
             'total_day_min': {"min": 0, "max": 500, "median": 182.10},
             'total_day_calls': {"min": 0, "max": 800, "median": 101},
             'total_day_charge': {"min": 0, "max": 61, "median": 30.91},
             'total_eve_min': {"min": 0, "max": 800, "median": 202.90},
             'total_eve_calls': {"min": 0, "max": 170, "median": 100},
             'total_eve_charge': {"min": 0, "max": 31, "median": 17.26},
             'total_night_minutes': {"min": 0, "max": 800, "median": 202.0},
             'total_night_calls': {"min": 0, "max": 175, "median": 100},
             'total_night_charge': {"min": 0, "max": 200, "median": 9.09},
             'total_intl_minutes': {"min": 0, "max": 25, "median": 10.30},
             'total_intl_calls': {"min": 0, "max": 20, "median": 4},
             'total_intl_charge': {"min": 0, "max": 10, "median": 2.78},
             'customer_service_calls': {"min": 0, "max": 10, "median": 1},
             "445.0": False,
             "452.0": False,
             "547.0": False,
             'total_charge': False,
             'total_calls': False,
             'total_min': False,
             'no_of_plans': False,
             'avg_call_mins': False}

required = [
    'account_length',
    'intertiol_plan',
    'voice_mail_plan',
    'number_vm_messages',
    'total_day_min',
    'total_day_calls',
    'total_day_charge',
    'total_eve_min',
    'total_eve_calls',
    'total_eve_charge',
    'total_night_minutes',
    'total_night_calls',
    'total_night_charge',
    'total_intl_minutes',
    'total_intl_calls',
    'total_intl_charge',
    'customer_service_calls',
    'location_code'
]


def preprocess_data(data):
    loc_code = data["location_code"]
    if(loc_code == "445"):
        data["445.0"] = 1
        data["452.0"] = 0
        data["547.0"] = 0
    elif(loc_code == "452"):
        data["445.0"] = 0
        data["452.0"] = 1
        data["547.0"] = 0
    elif(loc_code == "547"):
        data["445.0"] = 0
        data["452.0"] = 0
        data["547.0"] = 1

    if data["intertiol_plan"] == "Yes":
        data["intertiol_plan"] = 1
    elif data["intertiol_plan"] == "No":
        data["intertiol_plan"] = 0

    if data["voice_mail_plan"] == "Yes":
        data["voice_mail_plan"] = 1
    elif data["voice_mail_plan"] == "No":
        data["voice_mail_plan"] = 0

    data["total_charge"] = data["total_intl_charge"] + data["total_night_charge"] + \
        data["total_eve_charge"] + data["total_day_charge"]
    data["total_calls"] = data["total_intl_calls"] + data["total_night_calls"] + \
        data["total_eve_calls"] + data["total_day_calls"]
    data["total_min"] = data["total_intl_minutes"] + \
        data["total_night_minutes"] + \
        data["total_eve_min"] + data["total_day_min"]
    data["no_of_plans"] = data['intertiol_plan'] + data['voice_mail_plan']
    data["avg_call_mins"] = data["total_min"] / data["total_calls"]

    values = []
    for feature in x_columns.keys():
        column_values = x_columns[feature]
        if column_values and (data[feature] > column_values["max"] or data[feature] < column_values["min"]):
            print("Invalid value for feature: {}".format(feature))
            data[feature] = column_values["median"]

        values.append(data[feature])

    return values


@app.errorhandler(JsonValidationError)
def validation_error(e):
    return jsonify({'error': e.message, 'errors': [validation_error.message for validation_error in e.errors]})

# routes


@app.route('/')
def hello():
    return make_response(jsonify({
        'message': 'Hello World'
    }), 200)


predict_schema = {
    'required': required,
    'properties': {
        # 'todo': {'type': 'string'},
        # 'priority': {'type': 'integer'},
    }
}


@app.route('/api/v1/predict', methods=['POST'])
@schema.validate(predict_schema)
def predict():
    print("started")
    try:
        # print("x-")
        model = load('model.joblib')
        data = request.get_json()

        # print("pre-processing")
        values = preprocess_data(data)

        print(values)
        final_data = [np.array(values)]
        prediction = model.predict(final_data)
        # prediction = 'lept'
        print("prediction :", prediction[0])
        return make_response(jsonify({
            'prediction': str(prediction[0])
        }), 200)
    except Exception as e:
        return make_response(jsonify({
            'message': str(e)
        }), 500)


@app.route('/api/v1/customer', methods=['POST'])
def customer():
    try:
        data = request.get_json()
        # print(data)
        customer_id = data['customer_id']
        if(customer_id < 1000 or customer_id > 2319):
            return make_response(jsonify({
                'message': 'Invalid customer id'
            }), 400)

        df = pd.read_csv('web.csv')
        customer = df[df['customer_id'] == 1001].to_dict(orient='index')[0]
        # print(customer)
    except Exception as e:
        return make_response(jsonify({
            'message': str(e)
        }), 500)
    return make_response(jsonify({
        'message': 'Customer found',
        'data': customer
    }), 200)


if __name__ == "__main__":
    app.run(debug=True)
