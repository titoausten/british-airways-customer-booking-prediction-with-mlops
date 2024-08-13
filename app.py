from flask import Flask, request, render_template
from flask_cors import CORS,cross_origin
from src.pipeline.predict_pipeline import CustomData, PredictPipeline


application = Flask(__name__)
app = application

# Route for a home page
@app.route('/')
@cross_origin()
def index():
    return render_template('index.html')


@app.route('/predictdata', methods=['GET', 'POST'])
@cross_origin()
def predict_datapoint():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        data = CustomData(
            num_passengers=int(request.form.get('num_passengers')),
            sales_channel=request.form.get('sales_channel'),
            trip_type=request.form.get('trip_type'),
            purchase_lead=int(request.form.get('purchase_lead')),
            length_of_stay=int(request.form.get('length_of_stay')),
            flight_hour=int(request.form.get('flight_hour')),
            flight_day=request.form.get('flight_day'),
            route = request.form.get('route'),
            booking_origin = request.form.get('booking_origin'),
            wants_extra_baggage = int(request.form.get('wants_extra_baggage')),
            wants_preferred_seat = int(request.form.get('wants_preferred_seat')),
            wants_in_flight_meals = int(request.form.get('wants_in_flight_meals')),
            flight_duration = float(request.form.get('flight_duration'))
            )

        prediction_dataframe = data.get_data_as_dataframe()
        print(prediction_dataframe)
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(prediction_dataframe)

        if results[0] > 0.5:
            results = "Not Purchase"
        else:
            results = "Purchase"

        return render_template('index.html', results=results)
        #return render_template('index.html', results=results[0])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
    # app.run(host="0.0.0.0", port=80)  # For Azure Cloud
