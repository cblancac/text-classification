from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from transformers import Trainer, TrainingArguments

from dataset.load_data import LoadDataset
from dataset.tokenize_dataset import TokenizeDataset
from config import SentimentConfig
from model import SentimentClassifier


if __name__ == '__main__':

    model_ckpt = "distilbert-base-uncased"

    data_loader = LoadDataset('emotion')
    data = data_loader.load_data()

    tk = TokenizeDataset(model_ckpt)
    dataset_tokenized = tk.encoded_dataset(data)
    dataset_tokenized.set_format("torch", 
                        columns=["input_ids", "attention_mask", "label"])
    
    config = SentimentConfig(
        dataset_train=dataset_tokenized["train"],
        num_labels=6,
        model_ckpt=model_ckpt,
    )  

    model = SentimentClassifier(
        config=config.custom_config,
        model_ckpt=model_ckpt,
    )

batch_size = 64
logging_steps = len(dataset_tokenized["train"]) // batch_size
model_name = f"{model_ckpt}-finetuned-emotion"
training_args = TrainingArguments(output_dir=model_name,
                                num_train_epochs=2,
                                learning_rate=2e-5,
                                per_device_train_batch_size=batch_size,
                                per_device_eval_batch_size=batch_size,
                                weight_decay=0.01,
                                evaluation_strategy="epoch",
                                disable_tqdm=False,
                                logging_steps=logging_steps,
                                push_to_hub=False, 
                                log_level="error")
    
trainer = Trainer(model=model.model, args=training_args, 
                compute_metrics=model.compute_metrics,
                train_dataset=dataset_tokenized["train"],
                eval_dataset=dataset_tokenized["validation"],
                tokenizer=tk.tokenizer)

trainer.train()
trainer.save_model(model_name)
model.save_custom_configuration(model_name)

model.export_metrics(trainer, 
                        dataset_tokenized["test"],
                        f"{model_name}/test_metrics.csv")