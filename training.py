from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import joblib

print("Mulai training model...")


# 1. Load dataset Iris
iris = load_iris()
x = iris.data
y = iris.target
print("Dataset Iris berhasil dimuat.")  

# 2. Inisialisasi model KNN dengan k=3
model = KNeighborsClassifier(n_neighbors=3 )
print("Model KNN berhasil diinisialisasi.") 

# 3. Latih model dengan data Iris
model.fit(x, y)
print("Model berhasil dilatih.")    
# 4. Simpan model yang telah dilatih ke file
joblib.dump(model, 'model_iris.joblib')
print("Model berhasil disimpan ke 'model_iris.joblib'.")    

# Simpan nama target
joblib.dump(iris.target_names, 'nama_bunga.joblib')
print("Nama target berhasil disimpan ke 'model_iris.joblib'")  
print("Training selesai.")