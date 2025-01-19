#!/usr/bin/env python
# coding: utf-8

# ### Libs import

# #### Primary libs

# In[123]:


from tqdm.notebook import tqdm

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import optuna

import warnings
warnings.filterwarnings("ignore")


# #### ML modules

# Torch

# In[124]:


import torch
import torch.nn as nn
import torch.optim as optim

from torch.optim.lr_scheduler import StepLR


# In[125]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')


# Not Torch

# In[126]:


import gc

from scipy.signal import correlate
from statsmodels.graphics.tsaplots import plot_acf
from torch.utils.data import TensorDataset, DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Custom modules

# In[127]:


from modules.loss_functions import SharpeLoss, MeanReturnLoss
from modules.networks import LSTMModel, LSTMWithAttention, LSTMWithAttentionAlt, ConvLSTMWithAttention


# ### Functions definings

# ### Dataloading

# In[128]:


PATH = 'data/binance/fut/hour'

close = pd.read_csv(f'{PATH}/close.csv', index_col='openTime')
open = pd.read_csv(f'{PATH}/open.csv', index_col='openTime')
high = pd.read_csv(f'{PATH}/high.csv', index_col='openTime')
low = pd.read_csv(f'{PATH}/low.csv', index_col='openTime')
qvolume = pd.read_csv(f'{PATH}/qvolume.csv', index_col='openTime')
bvolume = pd.read_csv(f'{PATH}/bvolume.csv', index_col='openTime')
ntrades = pd.read_csv(f'{PATH}/ntrades.csv', index_col='openTime')
takerbuybvolume = pd.read_csv(f'{PATH}/takerbuybvolume.csv', index_col='openTime')
takerbuyqvolume = pd.read_csv(f'{PATH}/takerbuyqvolume.csv', index_col='openTime')


# In[129]:


good_tickers = close.isna().sum()[close.isna().sum() < 10_000].index

close = close[good_tickers] 
close.index = pd.to_datetime(close.index)
close = close.resample('4H').last()

open = open[good_tickers]
open.index = pd.to_datetime(open.index)
open = open.resample('4H').first()

high = high[good_tickers]
high.index = pd.to_datetime(high.index)
high = high.resample('4H').max()

low = low[good_tickers]
low.index = pd.to_datetime(low.index)
low = low.resample('4H').min()

bvolume = bvolume[good_tickers]
bvolume.index = pd.to_datetime(bvolume.index)
bvolume = bvolume.resample('4H').sum()

qvolume = qvolume[good_tickers]
qvolume.index = pd.to_datetime(qvolume.index)
qvolume = qvolume.resample('4H').sum()

ntrades = ntrades[good_tickers]
ntrades.index = pd.to_datetime(ntrades.index)
ntrades = ntrades.resample('4H').sum()

takerbuybvolume = takerbuybvolume[good_tickers]
takerbuybvolume.index = pd.to_datetime(takerbuybvolume.index)
takerbuybvolume = takerbuybvolume.resample('4H').sum()

takerbuyqvolume = takerbuyqvolume[good_tickers]
takerbuyqvolume.index = pd.to_datetime(takerbuyqvolume.index)
takerbuyqvolume = takerbuyqvolume.resample('4H').sum()



returns = close.pct_change().shift(-1).dropna()
ret_col = np.array(returns.columns) + '_ret'
returns.columns = ret_col

close_col = np.array(close.columns) + '_close'
close.columns = close_col

open_col = np.array(open.columns) + '_open'
open.columns = open_col

high_col = np.array(high.columns) + '_high'
high.columns = high_col

low_col = np.array(low.columns) + '_low'
low.columns = low_col

bvolume_col = np.array(bvolume.columns) + '_bvolume'
bvolume.columns = bvolume_col

qvolume_col = np.array(qvolume.columns) + '_qvolume'
qvolume.columns = qvolume_col

bvolume_col = np.array(bvolume.columns) + '_bvolume'
bvolume.columns = bvolume_col

ntrades_col = np.array(ntrades.columns) + '_ntrades'
ntrades.columns = ntrades_col

takerbuybvolume_col = np.array(takerbuybvolume.columns) + '_takerbuybvolume'
takerbuybvolume.columns = takerbuybvolume_col

takerbuyqvolume_col = np.array(takerbuyqvolume.columns) + '_takerbuyqvolume'
takerbuyqvolume.columns = takerbuyqvolume_col


train_columns = np.array([close_col, open_col, high_col, low_col, bvolume_col, qvolume_col, ntrades_col, takerbuybvolume_col, takerbuyqvolume_col])
train_columns = train_columns.reshape(train_columns.shape[0] * train_columns.shape[1],)


full_data = pd.concat([
  close.pct_change(), open.pct_change(), high.pct_change(), low.pct_change(), bvolume.pct_change(),
  qvolume.pct_change(), ntrades.pct_change(), takerbuybvolume.pct_change(), takerbuyqvolume.pct_change(), returns
], axis=1).dropna()
full_data.index = pd.to_datetime(full_data.index) # Не только pct_change (close - оставляем, open, high, low / close, volume - выбрать один и поделить на него)

training_data = full_data[train_columns]



window_size = 100

all_x = np.lib.stride_tricks.sliding_window_view(training_data.values, (window_size, training_data.shape[1]))[:, 0]
all_y = full_data[ret_col].iloc[99:].values


# In[130]:


# Формируем обучающую выборку
x_test = all_x[-850:].view()  # Все, кроме последних 1000
y_test = all_y[-850:].view()  # Все, кроме последних 1000

x = all_x[:-(850 + 120)].view()
y = all_y[:-(850 + 120)].view()

# Определяем длину валидационной выборки
val_len = 1000

# Определяем индексы для валидационной выборки
val_start = x.shape[0] // 2 - val_len // 2
val_end = val_start + val_len

# Формируем валидационную выборку
x_val = x[val_start:val_end]
y_val = y[val_start:val_end]

# Определяем отступы для тестовой выборки
offset = 110

# Индексы для тестовой выборки
test_start = max(0, val_start - offset)
test_end = min(x.shape[0], val_end + offset)

# Формируем тренировочную выборку
x_train = np.concatenate((x[:test_start], x[test_end:]), axis=0)
y_train = np.concatenate((y[:test_start], y[test_end:]), axis=0)

# Проверяем размеры выборок
print(f"train_x shape: {x_train.shape}")
print(f"train_y shape: {y_train.shape}")
print(f"val_x shape: {x_val.shape}")
print(f"val_y shape: {y_val.shape}")
print(f"test_x shape: {x_test.shape}")
print(f"test_y shape: {y_test.shape}")


# In[131]:


x_train = torch.Tensor(x_train)
x_train = np.where(np.isinf(x_train), np.nan, x_train)
x_train = np.where(np.isnan(x_train), np.nanmean(x_train, axis=0), x_train)
x_train = torch.Tensor(x_train)

y_train = torch.tensor(y_train, dtype=torch.float32)
y_train = np.where(np.isinf(y_train), np.nan, y_train)
y_train = np.where(np.isnan(y_train), np.nanmean(y_train, axis=0), y_train)
y_train = torch.Tensor(y_train)


x_val = torch.Tensor(x_val)
x_val = np.where(np.isinf(x_val), np.nan, x_val)
x_val = np.where(np.isnan(x_val), np.nanmean(x_val, axis=0), x_val)
x_val = torch.Tensor(x_val)

y_val = torch.tensor(y_val, dtype=torch.float32)
y_val = np.where(np.isinf(y_val), np.nan, y_val)
y_val = np.where(np.isnan(y_val), np.nanmean(y_val, axis=0), y_val)
y_val = torch.Tensor(y_val)

x_test = torch.Tensor(x_test)
x_test = np.where(np.isinf(x_test), np.nan, x_test)
x_test = np.where(np.isnan(x_test), np.nanmean(x_test, axis=0), x_test)
x_test = torch.Tensor(x_test)

y_test = torch.tensor(y_test, dtype=torch.float32)
y_test = np.where(np.isinf(y_test), np.nan, y_test)
y_test = np.where(np.isnan(y_test), np.nanmean(y_test, axis=0), y_test)
y_test = torch.tensor(y_test, dtype=torch.float32)

print("Shape of x_train:", x_train.shape)  # Ожидаемая форма: (3107, 50, 465)
print("Shape of y_train:", y_train.shape)  # Ожидаемая форма: (3107,)

print("Shape of x_train:", x_val.shape)  # Ожидаемая форма: (3107, 50, 465)
print("Shape of y_train:", y_test.shape)  # Ожидаемая форма: (3107,)

print("Shape of x_train:", x_test.shape)  # Ожидаемая форма: (3107, 50, 465)
print("Shape of y_train:", y_test.shape)  # Ожидаемая форма: (3107,)


# ### ML

# In[132]:


class TimeSeriesDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

# Создаем dataset и dataloader
train_dataset = TimeSeriesDataset(x_train, y_train)
val_dataset = TimeSeriesDataset(x_val, y_val)
test_dataset = TimeSeriesDataset(x_test, y_test)

batch_size = 700  # Выбираем подходящий размер батча

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# In[133]:


share_values = []
weights_penalties = []
turnover_penalties = []


# In[134]:


sharpe_loss = SharpeLoss()
returns_loss = MeanReturnLoss()


# In[135]:


PARAMS_DICT = {
    'input_size': x_train.shape[2],
    'hidden_size': 16 * 30,
    'hidden_size2': 16 * 30,
    'output_size':  y_train.shape[1],
    'learning_rate': 0.00025,
    'dropout1': 0.7,
    'dropout2': 0.7,
    'num_epochs': 100,
    'num_heads': 16,
    'conv_out_channels': 64,
    'conv_kernel_size': 9,
    'PRINT_PARAM': 1,
    'device': device,
    'batch_size': batch_size,
    'alpha': sharpe_loss.alpha,
    'epsilon': sharpe_loss.epsilon,
    'weight_penalty_factor': sharpe_loss.weight_penalty_factor,
    'optimizer': 'Adam',
    'weight_decay': 0,
}

pred_dict = {}
pred_dict['train'] = []
pred_dict['val'] = []
pred_dict['test'] = []

# Гиперпараметры
input_size = PARAMS_DICT['input_size']  # Количество факторов (465)
hidden_size = PARAMS_DICT['hidden_size']  # Размер скрытого слоя
hidden_size2 = PARAMS_DICT['hidden_size2'] #
output_size = PARAMS_DICT['output_size']   # Одно выходное значение
learning_rate = PARAMS_DICT['learning_rate']
num_epochs = PARAMS_DICT['num_epochs']
PRINT_PARAM = PARAMS_DICT['PRINT_PARAM']

# Инициализация модели, потерь и оптимизатора
model = ConvLSTMWithAttention(
    input_size,
    hidden_size,
    hidden_size2,
    output_size,
    dropout=PARAMS_DICT['dropout1'],
    dropout2=PARAMS_DICT['dropout2'],
    num_heads=PARAMS_DICT['num_heads'],
    conv_out_channels=PARAMS_DICT['conv_out_channels']
).to(device)
criterion = returns_loss.to(device)
optimizer = optim.NAdam(model.parameters(), lr=learning_rate, weight_decay=PARAMS_DICT['weight_decay'])
scheduler = StepLR(optimizer, step_size=15, gamma=0.25)

train_losses = []
val_losses = []
train_sharpe_ratios = []
val_sharpe_ratios = []
train_turnovers = []
val_turnovers = []

# Early stopping threshold для Val Sharpe Ratio
early_stopping_threshold = 0.1
best_val_sharpe = float('-inf')  # Инициализируем худшим возможным значением

for epoch in tqdm(range(num_epochs)):
    model.train()
    epoch_loss = 0
    epoch_sharpe = 0
    epoch_turnover = 0

    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        # Обнуляем градиенты
        optimizer.zero_grad()

        # Прямой проход (forward pass)
        outputs = model(batch_x)

        # Вычисление потерь
        loss, sharpe_ratio, turnover = criterion(outputs.squeeze(), batch_y.float())

        # Обратный проход (backward pass) для вычисления градиентов
        loss.backward()

        # Обновление весов
        optimizer.step()

        # Агрегация потерь за эпоху
        epoch_loss += loss.item()
        epoch_sharpe += sharpe_ratio
        epoch_turnover += turnover

    epoch_loss /= len(train_loader)
    epoch_sharpe /= len(train_loader)
    epoch_turnover /= len(train_loader)

    train_losses.append(epoch_loss)
    train_sharpe_ratios.append(epoch_sharpe)
    train_turnovers.append(epoch_turnover)

    # Валидация модели
    model.eval()
    val_loss = 0
    val_sharpe = 0
    val_turnover = 0

    with torch.no_grad():
        for val_x, val_y in val_loader:
            val_x, val_y = val_x.to(device), val_y.to(device)
            val_outputs = model(val_x)
            loss, sharpe_ratio, turnover = criterion(val_outputs.squeeze(), val_y.float())

            val_loss += loss.item()
            val_sharpe += sharpe_ratio
            val_turnover += turnover

    val_loss /= len(val_loader)
    val_sharpe /= len(val_loader)
    val_turnover /= len(val_loader)

    val_losses.append(val_loss)
    val_sharpe_ratios.append(val_sharpe)
    val_turnovers.append(val_turnover)

    # Обновление learning rate через scheduler
    scheduler.step()

    # Печать результатов
    if epoch % PRINT_PARAM == 0:
        print(f'Epoch [{epoch}/{num_epochs}], Loss: {epoch_loss:.10f}, Validation Loss: {val_loss:.10f}')
        print(f'Train Sharpe Ratio: {epoch_sharpe:.10f}, Val Sharpe Ratio: {val_sharpe:.10f}')
        print(f'Train Turnover Penalty: {epoch_turnover:.10f}, Val Turnover Penalty: {val_turnover:.10f}')
        print()

    # Early stopping на основе Val Sharpe Ratio
    if val_sharpe > best_val_sharpe:
        best_val_sharpe = val_sharpe  # Обновляем лучшее значение
    if val_sharpe >= early_stopping_threshold:
        print(f"Early stopping at epoch {epoch}, Val Sharpe Ratio reached {val_sharpe:.4f}")
        break  # Прерываем обучение, если достигли или превысили пороговое значение

# Визуализация потерь и компонентов функции потерь
plt.figure(figsize=(10, 5))
plt.plot(range(PRINT_PARAM, len(train_losses) + 1, PRINT_PARAM), train_losses, label='Train Loss')
plt.scatter(range(PRINT_PARAM, len(val_losses) + 1, PRINT_PARAM), val_losses, label='Validation Loss', color='r')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(range(PRINT_PARAM, len(train_sharpe_ratios) + 1, PRINT_PARAM), train_sharpe_ratios, label='Train Sharpe Ratio')
plt.scatter(range(PRINT_PARAM, len(val_sharpe_ratios) + 1, PRINT_PARAM), val_sharpe_ratios, label='Validation Sharpe Ratio', color='r')
plt.xlabel('Epoch')
plt.ylabel('Sharpe Ratio')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(range(PRINT_PARAM, len(train_turnovers) + 1, PRINT_PARAM), train_turnovers, label='Train Turnover')
plt.scatter(range(PRINT_PARAM, len(val_turnovers) + 1, PRINT_PARAM), val_turnovers, label='Validation Turnover', color='r')
plt.xlabel('Epoch')
plt.ylabel('Turnover')
plt.legend()
plt.show()


# In[136]:


# Создаем 2 строки по 3 графика в каждой (валидация и тренировка)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))  # 2 строки и 3 колонки

# Первая строка - Validation
# Первый график: Validation Loss
axes[0, 0].scatter(range(PRINT_PARAM, len(val_losses) + 1, PRINT_PARAM), val_losses, label='Validation Loss', color='r')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Validation Loss')
axes[0, 0].legend()

# Второй график: Validation Sharpe Ratio
axes[0, 1].scatter(range(PRINT_PARAM, len(val_sharpe_ratios) + 1, PRINT_PARAM), val_sharpe_ratios, label='Validation Sharpe Ratio', color='b')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Sharpe Ratio')
axes[0, 1].legend()

# Третий график: Validation Turnover
axes[0, 2].scatter(range(PRINT_PARAM, len(val_turnovers) + 1, PRINT_PARAM), val_turnovers, label='Validation Turnover', color='g')
axes[0, 2].set_xlabel('Epoch')
axes[0, 2].set_ylabel('Turnover')
axes[0, 2].legend()

# Вторая строка - Train
# Первый график: Train Loss
axes[1, 0].scatter(range(PRINT_PARAM, len(train_losses) + 1, PRINT_PARAM), train_losses, label='Train Loss', color='r')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Train Loss')
axes[1, 0].legend()

# Второй график: Train Sharpe Ratio
axes[1, 1].scatter(range(PRINT_PARAM, len(train_sharpe_ratios) + 1, PRINT_PARAM), train_sharpe_ratios, label='Train Sharpe Ratio', color='b')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Sharpe Ratio')
axes[1, 1].legend()

# Третий график: Train Turnover
axes[1, 2].scatter(range(PRINT_PARAM, len(train_turnovers) + 1, PRINT_PARAM), train_turnovers, label='Train Turnover', color='g')
axes[1, 2].set_xlabel('Epoch')
axes[1, 2].set_ylabel('Turnover')
axes[1, 2].legend()

# Отображаем все графики
plt.tight_layout()  # Для корректного отображения всех графиков без наложения
plt.show()


# In[137]:


fig, axes = plt.subplots(2, 3, figsize=(15, 10))

model.eval()
with torch.no_grad():
    train_predictions = model(x_train[-850:].to(device)).detach().cpu().numpy()

pred_dict['train'].append(train_predictions)

pred_df = pd.DataFrame(train_predictions, columns=close.columns)
alpha1 = pred_df.subtract(pred_df.mean(axis=1), axis=0)
alpha1 = alpha1.div(alpha1.abs().sum(axis=1), axis=0)
axes[0, 0].plot(
    (alpha1.iloc[-850:] * pd.DataFrame(y_train[-850:].detach().numpy(), columns=close.columns)).dropna().sum(axis=1).cumsum(),
    label='neutralized'
)
axes[0, 0].legend(loc='upper left')
axes[0, 0].set_title('train_profit')
axes[0, 0].set_xlabel('hours')
axes[0, 0].set_ylabel('profit')
axes[0, 0].grid(True)  # Добавление сетки

with torch.no_grad():
    val_predictions = model(x_val[-850:].to(device)).detach().cpu().numpy()

pred_dict['val'].append(val_predictions)

pred_df = pd.DataFrame(val_predictions, columns=close.columns)
alpha2 = pred_df.subtract(pred_df.mean(axis=1), axis=0)
alpha2 = alpha2.div(alpha2.abs().sum(axis=1), axis=0)
axes[0, 1].plot(
    (alpha2.iloc[-850:] * pd.DataFrame(y_val[-850:].detach().numpy(), columns=close.columns)).dropna().sum(axis=1).cumsum(),
    label='neutralized'
)
axes[0, 1].legend(loc='upper left')
axes[0, 1].set_title('val_profit')
axes[0, 1].set_xlabel('hours')
axes[0, 1].set_ylabel('profit')
axes[0, 1].grid(True)  # Добавление сетки

with torch.no_grad():
    test_predictions = model(x_test.to(device)).detach().cpu().numpy()

pred_dict['test'].append(test_predictions)

pred_df = pd.DataFrame(test_predictions, columns=close.columns)
alpha3 = pred_df.subtract(pred_df.mean(axis=1), axis=0)
alpha3 = alpha3.div(alpha3.abs().sum(axis=1), axis=0)
axes[0, 2].plot(
    (alpha3.iloc[-len(x_test):] * pd.DataFrame(y_test.detach().numpy(), columns=close.columns)).sum(axis=1).cumsum(),
    label='neutralized'
)
axes[0, 2].legend(loc='upper left')
axes[0, 2].set_title('test_profit')
axes[0, 2].set_xlabel('hours')
axes[0, 2].set_ylabel('profit')
axes[0, 2].grid(True)  # Добавление сетки

# Установим одинаковые пределы по оси y для всех графиков
min_value = min(
    (alpha1.iloc[-850:] * pd.DataFrame(y_train[-850:].detach().numpy(), columns=close.columns)).dropna().sum(axis=1).cumsum().min(),
    (alpha2.iloc[-850:] * pd.DataFrame(y_val[-850:].detach().numpy(), columns=close.columns)).dropna().sum(axis=1).cumsum().min(),
    (alpha3.iloc[-850:] * pd.DataFrame(y_test.detach().numpy(), columns=close.columns)).dropna().sum(axis=1).cumsum().min(),
)
max_value = max(
    (alpha1.iloc[-850:] * pd.DataFrame(y_train[-850:].detach().numpy(), columns=close.columns)).dropna().sum(axis=1).cumsum().max(),
    (alpha2.iloc[-850:] * pd.DataFrame(y_val[-850:].detach().numpy(), columns=close.columns)).dropna().sum(axis=1).cumsum().max(),
    (alpha3.iloc[-850:] * pd.DataFrame(y_test.detach().numpy(), columns=close.columns)).dropna().sum(axis=1).cumsum().max()
)

min_value = min_value * 1.08 if min_value <= 0 else min_value - 0.03 * abs(max_value)
max_value = max_value * 1.08 if max_value >= 0 else max_value + 0.03 * abs(max_value)

print(min_value, max_value)

for i in range(3):
    axes[0, i].set_ylim(min_value, max_value)  # Одинаковые пределы по оси y для верхних графиков

# Рассчитаем минимальные и максимальные значения для оси x
x_min = min(alpha1.diff().abs().dropna().sum(axis=1).min(),
            alpha2.diff().abs().dropna().sum(axis=1).min(),
            alpha3.diff().abs().dropna().sum(axis=1).min())

x_max = max(alpha1.diff().abs().dropna().sum(axis=1).max(),
            alpha2.diff().abs().dropna().sum(axis=1).max(),
            alpha3.diff().abs().dropna().sum(axis=1).max())

x_min = x_min * 1.08
x_max = x_max * 1.08

# Построение гистограмм
hist_values1, bin_edges1 = np.histogram(alpha1.diff().abs().dropna().sum(axis=1), bins=120)
hist_values2, bin_edges2 = np.histogram(alpha2.diff().abs().dropna().sum(axis=1), bins=120)
hist_values3, bin_edges3 = np.histogram(alpha3.diff().abs().dropna().sum(axis=1), bins=120)

# Максимальное значение гистограммы
max_value = max([np.max(hist_values1), np.max(hist_values2), np.max(hist_values3)])

axes[1, 0].hist(alpha1.diff().abs().dropna().sum(axis=1), bins=120)
axes[1, 1].hist(alpha2.diff().abs().dropna().sum(axis=1), bins=120)
axes[1, 2].hist(alpha3.diff().abs().dropna().sum(axis=1), bins=120)

# Одинаковые пределы для оси y на гистограммах
for i in range(3):
    axes[1, i].set_ylim(0, max_value * 1.1)

# Установим одинаковые пределы по оси x для всех гистограмм
for ax in axes[1, :]:
    ax.set_xlim([x_min, x_max])

# Добавление вертикальных линий среднего
mean_alpha1 = (alpha1.diff().abs().dropna().sum(axis=1)).mean()
mean_alpha2 = (alpha2.diff().abs().dropna().sum(axis=1)).mean()
mean_alpha3 = (alpha3.diff().abs().dropna().sum(axis=1)).mean()

axes[1, 0].axvline(mean_alpha1, color='r', linestyle='dashed', linewidth=1, label=f'Mean: {mean_alpha1:.2f}')
axes[1, 1].axvline(mean_alpha2, color='r', linestyle='dashed', linewidth=1, label=f'Mean: {mean_alpha2:.2f}')
axes[1, 2].axvline(mean_alpha3, color='r', linestyle='dashed', linewidth=1, label=f'Mean: {mean_alpha3:.2f}')

# Заголовки подграфиков
axes[1, 0].set_title('Train')
axes[1, 0].set_xlabel('Absolute Change in positions')

axes[1, 1].set_title('Validation')
axes[1, 1].set_xlabel('Absolute Change in positions')

axes[1, 2].set_title('Test')
axes[1, 2].set_xlabel('Absolute Change in positions')


# Сохраняем график
plt.tight_layout()
# fig.savefig(os.path.join(output_dir, 'results.png'))
plt.show()


# In[138]:


plt.plot(
    ((alpha3.iloc[-len(x_test):] * pd.DataFrame(y_test.detach().numpy(), columns=close.columns)).sum(axis=1) - 2 * 1e-4 * \
    (alpha3 - alpha3.shift()).abs().sum(axis=1)).cumsum(),
    label='neutralized'
)
plt.show()


# In[114]:


_ = plt.hist(alpha1.values.flatten(), bins=1000)
plt.show()

_ = plt.hist(alpha2.values.flatten(), bins=1000)
plt.show()

_ = plt.hist(alpha3.values.flatten(), bins=1000)
plt.show()


# In[83]:


alpha3 = alpha3.clip(-0.04, 0.04)
alpha3 = alpha3.subtract(alpha3.mean(axis=1), axis=0)
alpha3 = alpha3.div(alpha3.abs().sum(axis=1), axis=0)
# _ = plt.hist(alpha1.values.flatten(), bins=1000)

plt.plot(
    (alpha3.iloc[-850:] * pd.DataFrame(y_test[-850:].detach().numpy(), columns=close.columns)).dropna().sum(axis=1).cumsum(),
    label='test'
)

alpha2 = alpha2.clip(-0.04, 0.04)
alpha2 = alpha2.subtract(alpha2.mean(axis=1), axis=0)
alpha2 = alpha2.div(alpha2.abs().sum(axis=1), axis=0)
# _ = plt.hist(alpha1.values.flatten(), bins=1000)

plt.plot(
    (alpha2.iloc[-850:] * pd.DataFrame(y_val[-850:].detach().numpy(), columns=close.columns)).dropna().sum(axis=1).cumsum(),
    label='val'
)

alpha1 = alpha1.clip(-0.04, 0.04)
alpha1 = alpha1.subtract(alpha1.mean(axis=1), axis=0)
alpha1 = alpha1.div(alpha1.abs().sum(axis=1), axis=0)
# _ = plt.hist(alpha1.values.flatten(), bins=1000)

plt.plot(
    (alpha1.iloc[-850:] * pd.DataFrame(y_train[-850:].detach().numpy(), columns=close.columns)).dropna().sum(axis=1).cumsum(),
    label='train'
)

plt.legend()
plt.show()


# In[135]:


plt.plot(
    (alpha1.iloc[-850:] * pd.DataFrame(y_train[-850:].detach().numpy(), columns=close.columns)).dropna().sum(axis=1).cumsum(),
    label='neutralized'
)


# In[55]:


# truncate (clip)


# In[48]:


print(alpha1.diff().abs().sum(axis=1).mean() * 6, alpha2.diff().abs().sum(axis=1).mean() * 6, alpha3.diff().abs().sum(axis=1).mean() * 6)
print(alpha1.diff().abs().sum(axis=1).mean(), alpha2.diff().abs().sum(axis=1).mean(), alpha3.diff().abs().sum(axis=1).mean())


# In[42]:


alpha1.isna().sum().mean(), alpha2.isna().sum().mean(), alpha3.isna().sum().mean()


# In[42]:


(alpha3 * y_test.detach().numpy()).sum(axis=1).cumsum().plot(label='test')
(alpha2 * y_val[-850:].detach().numpy()).sum(axis=1).cumsum().plot(label='val')

plt.legend()
plt.show()


# ### Idea

# In[73]:


a = np.array([1, 4, 2, 3, 2, 1, -1, -3, -4, -5, -10])

b = a - a.mean()
b = b / np.abs(b).sum()

c = np.concatenate((a[a > 0] / a[a > 0].sum(), a[a < 0] / np.abs(a[a < 0]).sum()))

plt.plot(b, label='neut')
plt.plot(c, label='clever norm')
plt.legend()


# In[74]:


# Подсчет параметров
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Количество обучаемых параметров: {num_params}")


# In[136]:


torch.save(model, 'models/binance/4hours/model15.pth')


# In[482]:


torch.save(model.state_dict(), "models/binance/hour/sixth.pth")


# In[224]:


share_values_np = np.array([tensor.cpu().detach().numpy() for tensor in share_values])
weights_penalties_np = np.array([tensor.cpu().detach().numpy() for tensor in weights_penalties])
turnover_penalties_np = np.array([tensor.cpu().detach().numpy() for tensor in turnover_penalties])
# weights_penalties
# turnover_penalties


# In[225]:


metrics = pd.DataFrame({
  'sharpe_values': share_values_np,
  # 'weights_penalties': weights_penalties_np,
  'turnover_penalties': turnover_penalties_np,
})

(metrics['turnover_penalties'] / metrics['sharpe_values']).abs().mean()


# In[226]:


(metrics['turnover_penalties'] / metrics['sharpe_values']).mean()


# In[227]:


metrics


# In[84]:


import json


# In[87]:


with open('models/binance/4hours/model12.json', 'w') as f:
    json.dumps(PARAMS_DICT, f)


# In[91]:


PARAMS_DICT['device']

