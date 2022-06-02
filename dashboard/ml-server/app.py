# dependencies
from flask import Flask, request, jsonify, make_response
from joblib import load
import numpy as np


from flask_json_schema import JsonSchema, JsonValidationError
from flask import Response

app = Flask(__name__)
schema = JsonSchema(app)

x_columns = ['account_length',
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
             "445.0",
             "452.0",
             "547.0",
             'total_charge',
             'total_calls',
             'total_min',
             'no_of_plans',
             'avg_call_mins']

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
    'customer_service_calls'
]


def preprocess_data(data):
    if("445.0" not in data):
        data["445.0"] = 0
    if("452.0" not in data):
        data["452.0"] = 0
    if ("547.0" not in data):
        data["547.0"] = 0

    data["total_charge"] = data["total_intl_charge"] + data["total_night_charge"] + \
        data["total_eve_charge"] + data["total_day_charge"]
    data["total_calls"] = data["total_intl_calls"] + data["total_night_calls"] + \
        data["total_eve_calls"] + data["total_day_calls"]
    data["total_min"] = data["total_intl_minutes"] + \
        data["total_night_minutes"] + \
        data["total_eve_min"] + data["total_day_min"]
    data["no_of_plans"] = data['intertiol_plan'] + data['voice_mail_plan']
    data["avg_call_mins"] = data["total_min"] / data["total_calls"]

    return data


@app.errorhandler(JsonValidationError)
def validation_error(e):
    return jsonify({'error': e.message, 'errors': [validation_error.message for validation_error in e.errors]})

# routes


@app.route('/')
def hello():
    return make_response(jsonify({
        'message': 'Hello World'
    }), 200)


todo_schema = {
    'required': required,
    'properties': {
        # 'todo': {'type': 'string'},
        # 'priority': {'type': 'integer'},
    }
}


@app.route('/api/v1/predict', methods=['POST'])
@schema.validate(todo_schema)
def predict():
    try:
        model = load('model.joblib')
        data = request.get_json()
        # print(data)

        values = []

        pre_proc_data = preprocess_data(data)
        # print("pre processed data :", pre_proc_data)

        for feature in x_columns:
            if feature not in pre_proc_data:
                return make_response(jsonify({
                    'message': 'Missing feature :'+str(feature)
                }), 400)

            values.append(pre_proc_data[feature])
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


if __name__ == "__main__":
    app.run(debug=True)
