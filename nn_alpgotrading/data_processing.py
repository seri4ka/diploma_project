import numpy as np
import pandas as pd
import torch

from typing import List, Tuple, Union


def load_data(
        file_path: str
    ) -> Tuple[pd.DataFrame, ...]:
    """
    Загружает данные из CSV-файлов, расположенных в указанной директории.

    :param file_path: Путь к директории с файлами.
    :return: Кортеж DataFrame'ов: close, open, high, low, qvolume, bvolume, ntrades, takerbuybvolume, takerbuyqvolume.
    """
    close = pd.read_csv(f'{file_path}/close.csv', index_col='openTime')
    open_ = pd.read_csv(f'{file_path}/open.csv', index_col='openTime')
    high = pd.read_csv(f'{file_path}/high.csv', index_col='openTime')
    low = pd.read_csv(f'{file_path}/low.csv', index_col='openTime')
    qvolume = pd.read_csv(f'{file_path}/qvolume.csv', index_col='openTime')
    bvolume = pd.read_csv(f'{file_path}/bvolume.csv', index_col='openTime')
    ntrades = pd.read_csv(f'{file_path}/ntrades.csv', index_col='openTime')
    takerbuybvolume = pd.read_csv(f'{file_path}/takerbuybvolume.csv', index_col='openTime')
    takerbuyqvolume = pd.read_csv(f'{file_path}/takerbuyqvolume.csv', index_col='openTime')

    return close, open_, high, low, qvolume, bvolume, ntrades, takerbuybvolume, takerbuyqvolume


def good_tickers_chooser(
        data: pd.DataFrame
    ) -> List[str]:
    """
    Возвращает список тикеров с менее чем 10 000 пропущенных значений.

    :param data: DataFrame с данными по тикерам.
    :return: Список имён тикеров.
    """
    return data.isna().sum()[data.isna().sum() < 10_000].index.tolist()


class DataResampler:
    """
    Класс для ресемплинга временных рядов с использованием различных методов агрегации.
    """

    @staticmethod
    def resample_data(
        data: pd.DataFrame,
        tickers: List[str],
        resample_interval: str,
        method: str) -> pd.DataFrame:
        """
        Универсальная функция ресемплинга данных.

        :param data: Исходный DataFrame.
        :param tickers: Список тикеров для ресемплинга.
        :param resample_interval: Интервал ресемплинга (например, '1D', '1H').
        :param method: Метод агрегации ('last', 'first', 'min', 'max', 'sum').
        :return: DataFrame с ресемплированными данными.
        """
        data = data[tickers]
        data.index = pd.to_datetime(data.index)
        if method == 'last':
            return data.resample(resample_interval).last()
        elif method == 'first':
            return data.resample(resample_interval).first()
        elif method == 'min':
            return data.resample(resample_interval).min()
        elif method == 'max':
            return data.resample(resample_interval).max()
        elif method == 'sum':
            return data.resample(resample_interval).sum()
        else:
            raise ValueError(f"Unsupported method: {method}")


def pct_calculator(
        data: pd.DataFrame,
        is_returns: bool = False
    ) -> pd.DataFrame:
    """
    Вычисляет процентные изменения данных.

    :param data: Исходный DataFrame.
    :param is_returns: Если True, сдвигает данные для прогнозирования доходностей.
    :return: DataFrame с процентными изменениями.
    """
    if is_returns:
        return data.pct_change().shift(-1).dropna()
    return data.pct_change().dropna()


def columns_renamer(
        data: pd.DataFrame,
        postfix: str
    ) -> pd.DataFrame:
    """
    Добавляет постфикс к названиям колонок.

    :param data: Исходный DataFrame.
    :param postfix: Постфикс, который будет добавлен к названиям колонок.
    :return: DataFrame с обновлёнными названиями колонок.
    """
    data.columns = [f"{col}_{postfix}" for col in data.columns]
    return data


def get_training_columns(
        *columns: List[str]
    ) -> List[str]:
    """
    Формирует список колонок для обучения.

    :param columns: Списки колонок, которые нужно объединить.
    :return: Развёрнутый список всех колонок.
    """
    return np.concatenate(columns).tolist()


def prepare_data(
    close: pd.DataFrame,
    open_: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    bvolume: pd.DataFrame,
    qvolume: pd.DataFrame,
    ntrades: pd.DataFrame,
    takerbuybvolume: pd.DataFrame,
    takerbuyqvolume: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """
    Обрабатывает данные:
    - Вычисляет процентные изменения для указанных колонок.
    - Объединяет все данные в один DataFrame.
    - Устанавливает индекс в формате datetime.

    :return: DataFrame с обработанными данными.
    """
    processed_data = pd.concat([
        close.pct_change(),
        open_.pct_change(),
        high.pct_change(),
        low.pct_change(),
        bvolume.pct_change(),
        qvolume.pct_change(),
        ntrades.pct_change(),
        takerbuybvolume.pct_change(),
        takerbuyqvolume.pct_change(),
        returns
    ], axis=1).dropna()

    processed_data.index = pd.to_datetime(processed_data.index)
    return processed_data


def generate_windows(
        data: pd.DataFrame,
        target_columns: List[str],
        window_size: int
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Генерирует обучающие выборки (окна) для временных рядов.

    :param data: Исходный DataFrame.
    :param target_columns: Колонки с целевыми значениями.
    :param window_size: Размер окна.
    :return: Кортеж из X (матрица признаков) и Y (вектор целевых значений).
    """
    training_data = data.values
    x = np.lib.stride_tricks.sliding_window_view(training_data, (window_size, training_data.shape[1]))[:, 0]
    y = data[target_columns].iloc[window_size - 1:].values
    return x, y

def prepare_test_data(
        all_x: np.ndarray,
        all_y: np.ndarray,
        test_size: int = 850
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Формирует тестовую выборку.
    
    :param all_x: Все данные (матрица признаков).
    :param all_y: Все целевые значения.
    :param test_size: Размер тестовой выборки.
    :return: Тестовые данные x и y.
    """
    x_test = all_x[-test_size:].reshape(-1, all_x.shape[1])
    y_test = all_y[-test_size:].reshape(-1)
    return x_test, y_test


def prepare_train_data(
        all_x: np.ndarray,
        all_y: np.ndarray,
        val_len: int = 1000,
        offset: int = 110
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Формирует тренировочную и валидационную выборки.
    
    :param all_x: Все данные (матрица признаков).
    :param all_y: Все целевые значения.
    :param val_len: Размер валидационной выборки.
    :param offset: Смещение для тестовой выборки.
    :return: Тренировочные данные x и y, валидационные данные x и y.
    """
    # Определяем индексы для валидационной выборки
    val_start = all_x.shape[0] // 2 - val_len // 2
    val_end = val_start + val_len

    x_val = all_x[val_start:val_end].reshape(-1, all_x.shape[1])
    y_val = all_y[val_start:val_end].reshape(-1)

    # Определяем отступы для тестовой выборки
    test_start = max(0, val_start - offset)
    test_end = min(all_x.shape[0], val_end + offset)

    x_train = np.concatenate((all_x[:test_start], all_x[test_end:]), axis=0)
    y_train = np.concatenate((all_y[:test_start], all_y[test_end:]), axis=0)

    return x_train, y_train, x_val, y_val


def handle_infs_and_nans(
        data: np.ndarray
    ) -> np.ndarray:
    """
    Обрабатывает NaN и бесконечности в данных, заменяя их на средние значения по колонкам.
    
    :param data: Исходные данные.
    :return: Данные с заменёнными NaN и бесконечностями.
    """
    data = np.where(np.isinf(data), np.nan, data)
    data = np.where(np.isnan(data), np.nanmean(data, axis=0), data)
    return data


def convert_to_tensor(
        x: np.ndarray,
        y: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Преобразует данные в тензоры PyTorch.

    :param x: Признаки.
    :param y: Целевые значения.
    :return: Признаки и целевые значения в виде тензоров.
    """
    x_tensor = torch.Tensor(x)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    return x_tensor, y_tensor
