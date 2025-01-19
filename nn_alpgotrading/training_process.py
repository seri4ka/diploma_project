import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import Tuple

class TimeSeriesDataset(Dataset):
    def __init__(
            self,
            x_data: torch.Tensor,
            y_data: torch.Tensor
        ):
        self.x_data = x_data
        self.y_data = y_data

    def __len__(
            self
        ) -> int:
        return len(self.x_data)

    def __getitem__(
            self,
            idx: int
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x_data[idx], self.y_data[idx]


def create_dataloader(
        x_data: torch.Tensor,
        y_data: torch.Tensor,
        batch_size: int
    ) -> DataLoader:
    """
    Создаёт DataLoader для тренировочных, валидационных и тестовых данных.

    :param x_data: Признаки.
    :param y_data: Целевые значения.
    :param batch_size: Размер батча.
    :return: DataLoader для данных.
    """
    dataset = TimeSeriesDataset(x_data, y_data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)
