import os
import pandas as pd
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import flwr as fl
import sys

# Carregar e agregar dados de cada RSU
def load_and_aggregate_data(file_path):
    a = pd.read_csv(file_path, sep=',', header=None, usecols=[0, 1, 3], 
                    names=['step', 'road_id', 'c02_emission'], skiprows=[0])
    # Converte o tempo para datetime e agrupa por 1 segundo
    a['step'] = pd.to_datetime(a['step'], unit='s')
    
    # COnverte co2_emission para numérico, forçando erros a NaN e preenchendo NaN com 0 
    a['c02_emission'] = pd.to_numeric(a['c02_emission'], errors='coerce').fillna(0)
    
    #Agrupa por 1 segundo, somando apenas as colunas numẽricas
    a = a.groupby([pd.Grouper(key='step', freq='1s')])[['c02_emission']].sum().reset_index()

    return a

# Pré-processamento dos dados
def create_time_series(data, time_col, value_col):
    data = data.sort_values(by=time_col)
    data[value_col] = data[value_col].rolling(5).mean()
    data.dropna(inplace=True)
    return data[[time_col, value_col]].set_index(time_col)

# Preparar dados para LSTM
def prepare_lstm_data(series, n_steps):
    X, y = [], []
    horizon = 5
    series = pd.Series(series).rolling(60).mean()
    series.dropna(inplace=True)
    series = series.values
    
    for i in range(len(series)):
        end_ix = i + n_steps
        if end_ix + horizon > len(series)-1:
            break
        seq_x, seq_y = series[i:end_ix], series[end_ix + horizon]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# Definir a classe cliente FLWR
class RSUClient(fl.client.NumPyClient):
    def __init__(self, rsu_id, X_train, y_train, X_test, y_test):
        self.rsu_id = rsu_id
        self.model = self.create_lstm_model((X_train.shape[1], 1))
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
    
    def create_lstm_model(self, input_shape):
        model = Sequential()
        model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=input_shape))
        model.add(LSTM(50, activation='relu', input_shape=input_shape))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        return model

    def get_parameters(self, config):
        return self.model.get_weights()
    
    def set_parameters(self, parameters):
        self.model.set_weights(parameters)
    
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        start_time = time.process_time()
        self.model.fit(self.X_train, self.y_train, epochs=10, batch_size=64, verbose=0)
        end_time = time.process_time()
        return self.get_parameters({}), len(self.X_train), {'time': end_time - start_time, "rsu_id": self.rsu_id}
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        y_pred = self.model.predict(self.X_test)
        self.plot_predictions(self.X_test, self.y_test, y_pred)
        mse = np.square(y_pred.flatten() - self.y_test).mean()
        return mse, len(self.X_test), {"rsu_id": self.rsu_id}

    def plot_predictions(self, X_test, y_test, y_pred):
        SIZE = 17500  # Defina o tamanho da amostra para visualização
        plt.figure(figsize=(18, 6))
        plt.plot(list(range(SIZE)), y_test[:SIZE], label='Real', color='blue')
        plt.plot(y_pred.flatten()[:SIZE], label='Prevision', color='magenta')
        plt.plot(X_test[:, -1, 0][:SIZE], label='Last Step', color='green', linestyle='dashed')
        plt.legend()
        #plt.title('Real vs Previsão de Emissões de CO2')
        plt.xlabel('Time')
        plt.ylabel('CO2 Emission')
        #plt.savefig(f'real_vs_previsao_RSU_{self.rsu_id}.png')
        #plt.show()
        
        # Calcular e imprimir os MSEs ingênuo e do modelo
        naive_mse = np.square(X_test[:, -1, 0] - y_test).mean()
        model_mse = np.square(y_pred.flatten() - y_test).mean()
        print(f"MSE naive: {naive_mse}, MSE do Modelo: {model_mse}")

if __name__ == "__main__":
    # Verificar se o ID da RSU foi fornecido como argumento
    if len(sys.argv) != 2:
        print("Usage: python client.py <rsu_id>")
        sys.exit(1)

    rsu_id = sys.argv[1]
    folder_path = '/home/carnot/hiaac/data/most/rsus_most_csv/'
    file_path = os.path.join(folder_path, f'RSU_{rsu_id}.csv')

    data = load_and_aggregate_data(file_path)
    time_series_data = create_time_series(data, 'step', 'c02_emission')

    n_steps = 10
    scaler = MinMaxScaler()
    series = scaler.fit_transform(time_series_data.values).flatten()
    X, y = prepare_lstm_data(series, n_steps)

    split_index = int(len(X) * 0.65) # 65% para treino, visto a quantidade de dados.
    X_train, y_train = X[:split_index], y[:split_index]
    X_test, y_test = X[split_index:], y[split_index:]

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    client = RSUClient(rsu_id, X_train, y_train, X_test, y_test)
    fl.client.start_numpy_client(server_address="[::]:9999", client=client)

    # Avaliar o modelo final
    mse, num_examples, _ = client.evaluate(client.get_parameters({}), {})
