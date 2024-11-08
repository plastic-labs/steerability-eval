from typing import Type
from steerability_eval.dataset.base import BaseDataset
from steerability_eval.dataset.w5 import W5Dataset as W5Dataset
from steerability_eval.dataset.statements import StatementsDataset

def get_dataset_class(name: str) -> Type[BaseDataset]:
    if name == 'StatementsDataset':
        return StatementsDataset
    else:
        raise ValueError(f'Unknown dataset: {name}')
