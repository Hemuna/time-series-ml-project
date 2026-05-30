import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

import torch
import torch.nn as nn

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


# 1. 读取 sunspot 数据
data = sm.datasets.sunspots.load_pandas().data

years = data["YEAR"].values
sunspots = data["SUNACTIVITY"].values.reshape(-1, 1)


# 2. 标准化数据
scaler = MinMaxScaler(feature_range=(-1, 1))
sunspots_scaled = scaler.fit_transform(sunspots)


# 3. 用过去 lookback 年预测下一年
def create_dataset(series, lookback=12):
    X, y = [], []
    for i in range(len(series) - lookback):
        X.append(series[i:i + lookback])
        y.append(series[i + lookback])
    return np.array(X), np.array(y)


lookback = 12
X, y = create_dataset(sunspots_scaled, lookback)


# 4. 划分 train / test
train_size = int(len(X) * 0.8)

X_train = X[:train_size]
y_train = y[:train_size]

X_test = X[train_size:]
y_test = y[train_size:]

years_pred = years[lookback:]
years_train = years_pred[:train_size]
years_test = years_pred[train_size:]


# 5. 转成 PyTorch tensor
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)

X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32)


# 6. RNN 模型，ReLU activation
class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, output_size=1):
        super(RNNModel, self).__init__()

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            nonlinearity="relu"
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, hidden = self.rnn(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


# 7. LSTM 模型
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, output_size=1):
        super(LSTMModel, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, hidden = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


# 8. 训练函数
def train_model(model, X_train, y_train, epochs=1000, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()

        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 200 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.6f}")


# 9. 训练模型
torch.manual_seed(0)

rnn_model = RNNModel()
lstm_model = LSTMModel()

print("Training RNN...")
train_model(rnn_model, X_train_t, y_train_t)

print("Training LSTM...")
train_model(lstm_model, X_train_t, y_train_t)


# 10. 预测
rnn_model.eval()
lstm_model.eval()

with torch.no_grad():
    rnn_train_pred = rnn_model(X_train_t).numpy()
    rnn_test_pred = rnn_model(X_test_t).numpy()

    lstm_train_pred = lstm_model(X_train_t).numpy()
    lstm_test_pred = lstm_model(X_test_t).numpy()


# 11. 反标准化
y_train_real = scaler.inverse_transform(y_train)
y_test_real = scaler.inverse_transform(y_test)

rnn_train_real = scaler.inverse_transform(rnn_train_pred)
rnn_test_real = scaler.inverse_transform(rnn_test_pred)

lstm_train_real = scaler.inverse_transform(lstm_train_pred)
lstm_test_real = scaler.inverse_transform(lstm_test_pred)


# 12. 计算 RMSE
rnn_rmse = np.sqrt(mean_squared_error(y_test_real, rnn_test_real))
lstm_rmse = np.sqrt(mean_squared_error(y_test_real, lstm_test_real))

print("RNN Test RMSE:", rnn_rmse)
print("LSTM Test RMSE:", lstm_rmse)


# 13. 保存 RNN 图像
plt.figure(figsize=(10, 5))
plt.plot(years, sunspots, label="Sunspot data")
plt.plot(years_train, rnn_train_real, label="RNN train prediction")
plt.plot(years_test, rnn_test_real, label="RNN test prediction")
plt.title(f"RNN Prediction, Test RMSE = {rnn_rmse:.3f}")
plt.xlabel("Year")
plt.ylabel("Sunspot Activity")
plt.legend()
plt.tight_layout()
plt.savefig("rnn_prediction.png", dpi=300)
plt.show()


# 14. 保存 LSTM 图像
plt.figure(figsize=(10, 5))
plt.plot(years, sunspots, label="Sunspot data")
plt.plot(years_train, lstm_train_real, label="LSTM train prediction")
plt.plot(years_test, lstm_test_real, label="LSTM test prediction")
plt.title(f"LSTM Prediction, Test RMSE = {lstm_rmse:.3f}")
plt.xlabel("Year")
plt.ylabel("Sunspot Activity")
plt.legend()
plt.tight_layout()
plt.savefig("lstm_prediction.png", dpi=300)
plt.show()