# Flight Fare Prediction Web App :

  A web application that predicts flight fares using **historical data with Machine Learning** and **real-time data via Amadeus API**.  
Users can select origin, destination, airline, travel class, and departure date to get fare predictions and recommendations.

## Features :

 **Origin & Destination Selection**: Choose from 6 major Indian cities.  
 **Airline Selection**: Choose from popular airlines (Indigo, AirAsia, SpiceJet, Vistara, Air India, GO FIRST).  
 **Travel Class**: Economy or Business.  
 **Departure Date**: Pick any current or future date.  
 **Fare Prediction**:
    - Current & near-future dates (up to June 2026): Fetches real-time fare from Amadeus API.  
    - Future dates beyond June 2026 : Predicts fare using trained ML model (Random Forest).  
    -  Past dates are not supported for predictions. 
 **Recommendation**: Suggests whether to book now or wait based on fare comparison.
 

 **NOTE**:   The 6 major indian cities , airlines and travel class are from dataset. And the dataset were taken from kaggle which has more real hisorical data's. 
 

## How It Works :

1. **User Input** :
   - Select Origin & Destination city  
   - Select Airline  
   - Select Travel Class (Economy / Business)  
   - Pick Departure Date  

2. **Fare Prediction Logic** :  
     **Current / Near-Future Dates (up to June 2026)**:  
     - App fetches **real-time fares** from **Amadeus API**.  
     **Future Dates (beyond June 2026)**:  
     - App uses **ML model (Random Forest)** trained on historical data to **predict fare**.  
     **Past Dates**:  
     - Not supported for predictions,, app will show a message that it cannot calculate fare.  

3. **Recommendation** :
   - Compares ML prediction with real-time API fare (if available).
   - Suggests whether to book now or wait based on the fare comparison.

4. **Result Display** : 
   - Shows Origin, Destination, Airline, Class, Departure Date  
   - Shows ML predicted fare (for future dates beyond June 2026)  
   - Shows Real-time fare (for current/near-future dates up to June 2026)  
   - Provides a recommendation message
     

## Tech Stack :

 **Frontend**: HTML, Bootstrap 5  
 **Backend**: Python, Flask  
 **Machine Learning**: Random Forest Regressor (trained on historical flight fare data)  
 **API**: Amadeus Flight Offers API  
 **Data Handling**: pandas, numpy  
 **Model Persistence**: pickle  
 

## Setup Instructions

1. **Install required packages**  
pip install -r requirements.txt

## Amadeus key:
AMADEUS_API_KEY = "nE2tTeAEy7UivcIT7hLB244wwNZXkwCT"
AMADEUS_API_SECRET = "2bL9EhQVobgpZfRV"

## Run the Flask app:
python app.py

## Open your browser and navigate to:
http://127.0.0.1:5000/

**Developed by**:
Aadharshini S
AI | ML | Data Science Enthusiast
