from typing import List

from datasets import load_dataset
from datasets.dataset_dict import DatasetDict
from datasets.arrow_dataset import Dataset


class LoadDataset():
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name

    def load_data(self) -> DatasetDict:
        return load_dataset(self.dataset_name)      
    
    def split_train_val_test(self, data: DatasetDict) -> List[Dataset]:
        train       = data['train']
        validation  = data['validation']
        test        = data['test']
        return train, validation, test