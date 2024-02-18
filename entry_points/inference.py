from pathlib import Path
import sys
import os

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from lime.lime_text import LimeTextExplainer
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification


# Specify the path of the model
model_ckpt = Path("./distilbert-base-uncased-finetuned-emotion")

# Load the fine-tuned tokenizer and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModelForSequenceClassification.from_pretrained(model_ckpt).to(device)

class_names = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
explainer = LimeTextExplainer(class_names=class_names)


def inference(text: str) -> str:
    inputs = tokenizer(text, return_tensors="pt")

    inputs = {k:v.to(device) for k,v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1).tolist()[0]

    max_vale = max(predictions)
    idx = predictions.index(max_vale)
    return model.config.id2label[idx]

def predict(text: str) -> str:
    inputs = tokenizer(text, return_tensors="pt", padding=True)

    inputs = {k:v.to(device) for k,v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    probas = torch.nn.functional.softmax(outputs.logits, dim=-1).detach().numpy()

    return probas


if __name__ == "__main__":
    path_to_export = Path('results')
    text = "today I feel happy because I am on holiday"

    if not os.path.exists(path_to_export):
        os.makedirs(path_to_export)

    exp = explainer.explain_instance(text, predict, num_features=5, num_samples=1000)
    exp.save_to_file(path_to_export / 'lime.html')
    print(inference(text))

