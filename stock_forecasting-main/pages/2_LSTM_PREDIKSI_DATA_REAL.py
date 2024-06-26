import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import streamlit as st
import plotly.express as px
from datetime import datetime
import plotly.graph_objects as go

st.set_page_config(
    layout='wide',
    page_title='LSTM FORECAST'
)

# Fungsi untuk membuat model LSTM
def buat_model_LSTM(time_step, epochs, batch_size, optimizer):
    # Langkah 1: Memuat Data dari Yahoo Finance
    data = yf.download('KKGI.JK', start='2020-01-01', end=datetime.now())

    # Langkah 2: Praproses Data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data[['Close']])

    # Membagi data menjadi set pelatihan dan pengujian (80% latih, 20% uji)
    train_size = int(len(data_scaled) * 0.8)
    train, test = data_scaled[:train_size], data_scaled[train_size:]

    # Langkah 3: Membuat Dataset
    def buat_dataset(dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - time_step):
            a = dataset[i:(i + time_step), 0]
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return pd.DataFrame(dataX), pd.Series(dataY)

    X_train, y_train = buat_dataset(train, time_step)
    X_test, y_test = buat_dataset(test, time_step)

    # Mengubah bentuk input menjadi [samples, time steps, features] yang diperlukan untuk LSTM
    X_train = X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Langkah 4: Membangun Model LSTM
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(1))
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    # Langkah 5: Melatih Model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, verbose=0)

    # Langkah 6: Membuat Prediksi
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    # Mengubah kembali ke nilai asli
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)

    # Menyiapkan data untuk disimpan ke CSV
    real_data = scaler.inverse_transform(data_scaled).flatten()
    train_data = pd.Series(index=data.index, data=[float('NaN')] * len(data))
    train_data[time_step:len(train_predict) + time_step] = train_predict.flatten()
    test_data = pd.Series(index=data.index, data=[float('NaN')] * len(data))
    test_data[len(train_predict) + (time_step * 2):len(data_scaled)] = test_predict.flatten()

    df = pd.DataFrame({
        'Tanggal': data.index,
        'Data Real': real_data,
        'Prediksi Latih': train_data.values,
        'Prediksi Uji': test_data.values,
        'Open': data['Open'],
        'High': data['High'],
        'Low': data['Low'],
        'Close': data['Close']
    })

    return model, df

# Formulir input parameter di sidebar
with st.sidebar.form(key='params_form'):
    st.markdown("<div style='display: flex; justify-content: center;'><h2>Perbandingan Data Real dan Prediksi LSTM</h2></div>", unsafe_allow_html=True)
    optimizers = ['adam', 'adamax', 'sgd', 'rmsprop']
    optimizer = st.selectbox('Optimizer', optimizers, key='optimizer_selectbox')
    time_step = st.number_input('Lookback', min_value=1, max_value=500, value=164, step=1)
    epochs_value = st.number_input('Epochs', min_value=1, value=100)
    batch_size_value = st.number_input('Batch Size', min_value=1, value=32)
    train_button = st.form_submit_button('Train Model')

# Konten utama
if train_button:
    # Membuat Model
    model, df = buat_model_LSTM(time_step, epochs_value, batch_size_value, optimizer)

    # Menghitung kesalahan antara nilai real dan prediksi
    df['Kesalahan Latih'] = df.apply(lambda row: abs(row['Data Real'] - row['Prediksi Latih']) if pd.notnull(row['Prediksi Latih']) else None, axis=1)
    df['Kesalahan Uji'] = df.apply(lambda row: abs(row['Data Real'] - row['Prediksi Uji']) if pd.notnull(row['Prediksi Uji']) else None, axis=1)

    # Menghitung rata-rata kesalahan
    rata_kesalahan_latih = df['Kesalahan Latih'].mean()
    rata_kesalahan_uji = df['Kesalahan Uji'].mean()

    st.success('Pelatihan model selesai!')

    z1, z2 = st.columns((8, 2.5))
    with z1:
        # Grafik garis untuk perbandingan data
        fig = px.line(df, x='Tanggal', y=['Data Real', 'Prediksi Latih', 'Prediksi Uji'], title='Data Real dan Hasil Prediksi')
        st.plotly_chart(fig)

        # Grafik garis dengan scatter plot dan area
        fig_combined = go.Figure()
        fig_combined.add_trace(go.Scatter(x=df['Tanggal'], y=df['Data Real'], mode='lines', name='Data Real', fill='tozeroy'))
        fig_combined.add_trace(go.Scatter(x=df['Tanggal'], y=df['Prediksi Latih'], mode='lines', name='Prediksi Latih', fill='tonexty'))
        fig_combined.add_trace(go.Scatter(x=df['Tanggal'], y=df['Prediksi Uji'], mode='lines', name='Prediksi Uji', fill='tonexty'))
        fig_combined.update_layout(
            title='Perbandingan Data Real dan Prediksi',
            xaxis_title='Tanggal',
            yaxis_title='Harga',
            legend_title='Legenda',
            template='plotly_white'
        )
     
