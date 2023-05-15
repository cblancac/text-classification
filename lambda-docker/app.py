import json
import requests
from pathlib import Path

import regex as re
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

from exceptions import BadContentType, EndpointError, BadRequest
import os
os.environ['TRANSFORMERS_CACHE'] = "./distilbert-base-uncased-finetuned-emotion"

# Specify the path of the model
model_ckpt = Path("./distilbert-base-uncased-finetuned-emotion")

# Load the fine-tuned tokenizer and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModelForSequenceClassification.from_pretrained(model_ckpt).to(device)


def inference(text: str) -> str:
    """This function take a text as an input and
       predict whether or not it is a header. The input
       has to be line extracted by tesseract"""
    print("TTTTTTTTTTTTT: ", text)
    inputs = tokenizer(text, return_tensors="pt")

    inputs = {k:v.to(device) for k,v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1).tolist()[0]

    max_vale = max(predictions)
    idx = predictions.index(max_vale)
    return model.config.id2label[idx]


def get_text(event, context):
    # Get the text sent by Postman
    raw_string = r'{}'.format(event['body'])
    body = json.loads(raw_string)
    original_text = body['text']
    return original_text


def handler(event, context):
    text = get_text(event, context)
    emotion_detected = inference(text)
    final = {'output': emotion_detected}

    return {
        'statusCode': 200,
        'body': json.dumps(final)
    }

