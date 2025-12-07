from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

#1. Inisialisasi aplikasi Flask
app = Flask(__name__)

#2. Load "otak beku" (mode) & nama bunga
model = joblib.load('model_iris.joblib')
nama_bunga = joblib.load('nama_bunga.joblib')   

#3. Definisikan "rute" (alamat URL) untuk halaman utama
#rute 1: halaman utama
@app.route('/')
def home():
    return render_template('index.html')

#rute 2: Alamat untuk "menebak"
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Ambil data JSON dari permintaan POST

    # Ubah data dari formulir menjadi format yang dimengerti model
    # Model kita dilatih dengan array 2D, jadi kita buat [[]]
    features = [np.array([
        data['sl'],
        data['sw'],
        data['pl'],
        data['pw']
    ])]

    #4. Lakukan prediksi
    hasil_prediksi = model.predict(features) #Hasilnya berupa angka

    #ambil nama bunga berdasarkan hasil prediksi
    hasil_prediksi_nama = nama_bunga[hasil_prediksi][0]

    #5 Kirim kembali hasilnya (sebagai JSON) ke website
    return jsonify({
        'prediksi' : hasil_prediksi_nama
    })
#6. Jalankan aplikasi Flask
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000) # Jalankan aplikasi
