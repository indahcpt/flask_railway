from flask import Flask, render_template, request
import numpy as np
import pickle
import pandas as pd

app = Flask(__name__)

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load label encoders
with open('encoder_mapping.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

# Load scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil data dari form
        data = {
            'Temperature': float(request.form['Temperature']),
            'Humidity': float(request.form['Humidity']),
            'Wind Speed': float(request.form['Wind_Speed']),
            'Precipitation': float(request.form['Precipitation']),
            'Cloud Cover': request.form['Cloud_Cover'],
            'Atmospheric Pressure': float(request.form['Atmospheric_Pressure']),
            'UV Index': float(request.form['UV_Index']),
            'Season': request.form['Season'],
            'Visibility': float(request.form['Visibility']),            
            'Location': request.form['Location']
        }

        # Encode kategori
        for col in ['Cloud Cover', 'Season', 'Location']:
            encoder = label_encoders[col]
            data[col] = encoder.transform([data[col]])[0]

        # Buat DataFrame untuk prediksi
        df_input = pd.DataFrame([data])

        # Normalisasi input dengan scaler yang sudah dilatih
        df_scaled = scaler.transform(df_input)

        # Prediksi dengan model
        prediction = model.predict(df_scaled)[0]

        return render_template('index.html', predict_text=f'{prediction}')
    
    except Exception as e:
        return render_template('index.html', predict_text=f'Error: {str(e)}')

# Tambahan: route lain seperti 'about' dan 'tree'
@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/tree')
def tree():
    # Kalau ada visualisasi decision tree, kamu bisa generate di sini
    return render_template('tree.html')

if __name__ == '__main__':
    app.run(debug=True)
