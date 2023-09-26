import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Leggi i dati dal tuo file Excel
df1= pd.read_excel('GW_02.xlsx')
df = df1[(df1["Years"] >= 1980) & (df1["Years"] <= 2019)]

# Estrai le colonne necessarie
X = df[['GDP', 'Total population by sex', 'MtCO2e']]
y = df['Annual Mean']

# Normalizza i dati
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
y = scaler.fit_transform(y.values.reshape(-1, 1))

# Dividi i dati in set di addestramento e set di test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Costruisci il modello RNN
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, input_shape=(X_train.shape[1], 1), return_sequences=True),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(1)
])

# Compila il modello
model.compile(optimizer='adam', loss='mean_squared_error')

# Addestra il modello
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Visualizza le curve di apprendimento
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Prevedi i valori futuri
future_years = np.arange(2020, 2031).reshape(-1, 1)
future_predictions = model.predict(scaler.transform(future_years))

# Denormalizza le previsioni
future_predictions = scaler.inverse_transform(future_predictions)

# Crea un DataFrame per i dati futuri
future_df = pd.DataFrame({'Years': future_years.flatten(), 'Predicted_Annual_Mean': future_predictions.flatten()})
print(future_df)