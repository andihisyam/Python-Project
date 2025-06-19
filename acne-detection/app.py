
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Konfigurasi halaman
st.set_page_config(
    page_title="Dashboard Prediksi Konsentrasi CO(GT)",
    page_icon="🌫️"
)

st.title("Dashboard Prediksi Konsentrasi Karbon Monoksida (CO-GT)")
st.write("Aplikasi ini memprediksi konsentrasi CO (GT) berdasarkan data kualitas udara.")

st.markdown("---")
st.subheader("Evaluasi Model")

# 1. Load Model
rf_model = joblib.load("model.pkl")

# 2. Load dan persiapkan data
data = pd.read_csv('/content/drive/MyDrive/Data Mining/AirQualityUCI.csv', sep=';', decimal=',')
data.replace(-200, np.nan, inplace=True)
data.drop(columns=["Date", "Time", "Unnamed: 15", "Unnamed: 16"], inplace=True, errors="ignore")
data.dropna(inplace=True)

# 3. Siapkan fitur dan target
features = ['PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)', 'PT08.S2(NMHC)',
            'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)',
            'PT08.S5(O3)', 'T', 'RH', 'AH']
X = data[features]
y = data["CO(GT)"]

# 4. Prediksi dan evaluasi
y_pred_rf = rf_model.predict(X)  # pakai X, bukan X_test yang gak ada

mae = mean_absolute_error(y, y_pred_rf)
mse = mean_squared_error(y, y_pred_rf)
r2 = r2_score(y, y_pred_rf)

# Tampilkan hasil evaluasi dalam tabel
eval_df = pd.DataFrame({
    "Model": ["Random Forest Regression"],
    "MSE": [mse],
    "MAE": [mae],
    "R-squared": [r2]
})

st.dataframe(eval_df.style.format({
    "MSE": "{:.6f}",
    "MAE": "{:.6f}",
    "R-squared": "{:.6f}"
}))

# 5. Visualisasi aktual vs prediksi
st.subheader("Visualisasi Aktual vs Prediksi")
fig, ax = plt.subplots()
ax.scatter(y, y_pred_rf, alpha=0.3, color="steelblue")
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
ax.set_xlabel("CO(GT) Aktual")
ax.set_ylabel("CO(GT) Prediksi")
ax.set_title("Scatter Plot: Aktual vs Prediksi")
st.pyplot(fig)

st.markdown("---")
st.subheader("Prediksi CO(GT) Berdasarkan Input Manual")

# 6. Form Input Manual
with st.form("prediction_form"):
    input_data = {}
    for feature in features:
        input_data[feature] = st.number_input(feature, value=float(X[feature].mean()))
    submit = st.form_submit_button("Prediksi")

if submit:
    input_df = pd.DataFrame([input_data])
    try:
        prediction = rf_model.predict(input_df)[0]  # harus rf_model
        st.success(f"Prediksi Konsentrasi CO(GT): **{prediction:.2f} mg/m³**")
    except Exception as e:
        st.error(f"Gagal melakukan prediksi: {e}")
