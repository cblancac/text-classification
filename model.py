import os

import pandas as pd

import torch
from transformers import AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score

class SentimentClassifier():
    def __init__(
            self, 
            config,
            model_ckpt: str = "",
            ):
        self.config     = config
        self.model_ckpt = model_ckpt
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model      = AutoModelForSequenceClassification.from_pretrained(self.model_ckpt, config=self.config).to(self.device)
        
                
    def export_metrics(self, trainer, dataset_test, output_path):
        preds_output = trainer.predict(dataset_test)
        preds_output = {metric: [value] for (metric,value) in preds_output.metrics.items()}
        df_metrics = pd.DataFrame(preds_output)
        df_metrics.to_csv(output_path, index = False)
        
    def compute_metrics(self, pred):
        labels = pred.label_ids
        preds  = pred.predictions.argmax(-1)
        f1     = f1_score(labels, preds, average="weighted")
        acc    = accuracy_score(labels, preds)
        return {"accuracy": acc, "f1": f1}

    def save_custom_configuration(self, output_path_config):
        if not os.path.exists(output_path_config):
            os.makedirs(output_path_config)
        self.model.config.save_pretrained(output_path_config)

    def _associate_label_to_categories(self):
        labels = self.dataset_train.features["label"].names
        index2tag = {id: label for (id, label) in enumerate(labels)}
        tag2index = {label: id for (id, label) in index2tag.items()}
        return index2tag, tag2index

