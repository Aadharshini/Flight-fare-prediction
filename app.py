from flask import Flask, request, render_template
import pickle
import numpy as np
import requests
from datetime import datetime

app = Flask(__name__)

# Load model + encoders
with open("fare_model.pkl", "rb") as f:
    model, label_encoders = pickle.load(f)

# Amadeus API keys
AMADEUS_API_KEY = "nE2tTeAEy7UivcIT7hLB244wwNZXkwCT"
AMADEUS_API_SECRET = "2bL9EhQVobgpZfRV"

# City & Airline Mappings
airports = {
    "Chennai": "MAA",
    "Delhi": "DEL",
    "Mumbai": "BOM",
    "Bangalore": "BLR",
    "Hyderabad": "HYD",
    "Kolkata": "CCU"
}

airlines = {
    "Indigo": {"ml_name": "Indigo", "iata": "6E"},
    "AirAsia": {"ml_name": "AirAsia", "iata": "I5"},
    "SpiceJet": {"ml_name": "SpiceJet", "iata": "SG"},
    "Vistara": {"ml_name": "Vistara", "iata": "UK"},
    "Air_India": {"ml_name": "Air_India", "iata": "AI"},
    "GO_FIRST": {"ml_name": "GO_FIRST", "iata": "G8"}
}

classes = ["Economy", "Business"]

# Amadeus Token
def get_amadeus_token():
    url = "https://test.api.amadeus.com/v1/security/oauth2/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {
        "grant_type": "client_credentials",
        "client_id": AMADEUS_API_KEY,
        "client_secret": AMADEUS_API_SECRET
    }
    response = requests.post(url, headers=headers, data=data)
    return response.json().get("access_token")

# Fetch fare from Amadeus
def fetch_real_fare(origin, destination, date, travel_class):
    token = get_amadeus_token()
    url = "https://test.api.amadeus.com/v2/shopping/flight-offers"
    headers = {"Authorization": f"Bearer {token}"}
    params = {
        "originLocationCode": origin,
        "destinationLocationCode": destination,
        "departureDate": date,
        "adults": 1,
        "travelClass": travel_class.upper(),
        "currencyCode": "INR",
        "max": 5
    }
    response = requests.get(url, headers=headers, params=params)
    data = response.json()
    try:
        return data["data"][0]["price"]["total"]
    except Exception:
        return None

# Routes
@app.route("/")
def home():
    today = datetime.today().strftime("%Y-%m-%d")
    return render_template(
        "index.html",
        airports=airports,
        airlines=airlines.keys(),
        classes=classes,
        today=today
    )

@app.route("/predict", methods=["POST"])
def predict():
    origin_city = request.form["origin"]
    dest_city = request.form["destination"]
    selected_airline = request.form["airline"] 
    travel_class = request.form["travel_class"]
    date_str = request.form["date"]

    origin = airports[origin_city]
    destination = airports[dest_city]
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")

    prediction, real_price = None, None

    # Use API for dates up to June 30, 2026
    if date_obj <= datetime(2026, 6, 30):
        real_price = fetch_real_fare(origin, destination, date_str, travel_class)
    else:
        try:
            # Normalize ML features
            airline_norm = airlines[selected_airline]["ml_name"]
            travel_class_norm = travel_class.title()

            origin_encoded = label_encoders["source_city"].transform([origin_city])[0]
            destination_encoded = label_encoders["destination_city"].transform([dest_city])[0]
            airline_encoded = label_encoders["airline"].transform([airline_norm])[0]
            class_encoded = label_encoders["class"].transform([travel_class_norm])[0]

            # Use known dummy values from training to avoid unseen label error
            dep_time_encoded = label_encoders["departure_time"].transform([label_encoders["departure_time"].classes_[0]])[0]
            arr_time_encoded = label_encoders["arrival_time"].transform([label_encoders["arrival_time"].classes_[0]])[0]
            stops_encoded = label_encoders["stops"].transform([label_encoders["stops"].classes_[0]])[0]

            # Build input array for ML
            input_data = np.array([[airline_encoded, origin_encoded,
                                    destination_encoded, class_encoded,
                                    date_obj.month, date_obj.day,
                                    dep_time_encoded, arr_time_encoded, stops_encoded]])
            prediction = model.predict(input_data)[0]
        except Exception as e:
            prediction = f"Encoding error: {str(e)}"

    # Recommendation logic
    if prediction and real_price:
        if float(real_price) < float(prediction):
            recommendation = "Recommended: Book now! Real-time fare is lower than ML prediction."
        else:
            recommendation = "Predicted fare is lower. You may wait if flexible."
    elif real_price:
        recommendation = "Showing real-time fare. Recommended to book soon."
    elif prediction:
        recommendation = "Showing ML predicted fare (future date). Use as guidance."
    else:
        recommendation = "Could not calculate fare. Try changing inputs."

    return render_template(
        "results.html",
        origin=origin_city,
        destination=dest_city,
        airline=selected_airline,
        travel_class=travel_class,
        date=date_str,
        prediction=prediction if prediction else None,
        real_price=real_price,
        recommendation=recommendation
    )

if __name__ == "__main__":
    app.run(debug=True)
