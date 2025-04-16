from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Model ve scaler dosya yollarını ayarla
model_path = 'C:/Users/iremm/AI-IK/app/svm_model.pkl'
scaler_path = 'C:/Users/iremm/AI-IK/app/scaler.pkl'

# Model ve scaler'ı yükle
try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    print("Model ve scaler başarıyla yüklendi.")
except Exception as e:
    print(f"Model veya scaler yüklenirken hata oluştu: {e}")

@app.route("/", methods=["GET", "POST"])
def index():
    result = None  # Tahmin sonucunu burada saklayacağız
    if request.method == "POST":
        try:
            # Formdan gelen verileri al
            tecrube = float(request.form["tecrube"])
            puan = float(request.form["puan"])

            print(f"Tecrübe: {tecrube}, Puan: {puan}")  # Verileri terminalde kontrol et

            # Model tahmini yapma kısmı
            features = np.array([tecrube, puan]).reshape(1, -1)
            scaled_features = scaler.transform(features)
            prediction = model.predict(scaled_features)[0]

            print(f"Tahmin Sonucu: {prediction}")  # Tahmin sonucunu kontrol et

            # Tahmin sonucunu sakla
            result = prediction

            # Sonucu render_template ile şablona gönder
            return render_template("index.html", result=result, tecrube=tecrube, puan=puan)

        except Exception as e:
            print(f"Hata oluştu: {str(e)}")  # Hata mesajını terminalde gör
            return f"Hata oluştu: {str(e)}"
    
    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)
