from flask import Flask
from flask import request

app = Flask(__name__)
session = {}


@app.route("/", methods=["GET"])
def handle_request():
    session['vals'] = request.args.to_dict()
    print(session['vals'])

    return "Received"


@app.route("/data", methods=["GET"])
def return_data():
    if 'vals' in session:
        return f"{session['vals']['base']} {session['vals']['green']} {session['vals']['yellow']} " \
               f"{session['vals']['rigid']} {session['vals']['misc']}"
    else:
        return "None"
