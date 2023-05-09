from transformers import AutoConfig

class SentimentConfig():
    def __init__(
            self, 
            dataset_train,
            num_labels: int = 6,
            model_ckpt: str = "",
            
            ):
        self.dataset_train = dataset_train
        self.num_labels    = num_labels
        self.model_ckpt    = model_ckpt
        self.custom_config        = self.get_config()

    def get_config(self):
        index2tag, tag2index = self._associate_label_to_categories()
        return AutoConfig.from_pretrained(self.model_ckpt, 
                                           num_labels=self.num_labels,
                                           id2label = index2tag,
                                           label2id = tag2index)
        
    def _associate_label_to_categories(self):
        labels = self.dataset_train.features["label"].names
        index2tag = {id: label for (id, label) in enumerate(labels)}
        tag2index = {label: id for (id, label) in index2tag.items()}
        return index2tag, tag2index


