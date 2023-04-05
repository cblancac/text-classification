from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from dataset.load_data import LoadDataset
from dataset.tokenize_dataset import TokenizeDataset

if __name__ == '__main__':
    data_loader = LoadDataset('emotion')
    data = data_loader.load_data()

    tk = TokenizeDataset()
    dataset_tokenized = tk.encoded_dataset(data)