import numpy as np
import pandas as pd
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
