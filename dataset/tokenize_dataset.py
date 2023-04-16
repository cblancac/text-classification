from transformers import AutoTokenizer


class TokenizeDataset():
    def __init__(self, model_ckpt):
        self.model_ckpt = model_ckpt
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_ckpt)

    def encoded_dataset(self, dataset):
        return dataset.map(self._tokenize, batched=True, batch_size=None)

    def _tokenize(self, batch):
        return self.tokenizer(batch['text'], padding=True, truncation=True)

        