import json
import requests
from pathlib import Path

import regex as re
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

from exceptions import BadContentType, EndpointError, BadRequest


# Specify the path of the model
model_ckpt = Path("/opt/ml/model")

# Load the fine-tuned tokenizer and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModelForSequenceClassification.from_pretrained(model_ckpt).to(device)


def _get_json(request_body):
    print(type(request_body))
    try:
        decoded_request = request_body.decode("utf8").replace("'", '"')
    except AttributeError:
        decoded_request = request_body.replace("'", '"')
    try:
        query_value = re.search("(?<={\s*\"query\":\s*\")(.)+(?=\s*\"\s*}$)", decoded_request).group(0)
        query_value_fixed = query_value.replace('"', "")
        decoded_request_fixed = decoded_request.replace(query_value, query_value_fixed)
        print("-------------------- REQUEST BODY --------------------")
        print(f"TYPE of decoded request is: {type(decoded_request)}")
        print(f"CONTENT of decoded request is: {decoded_request}")
        print("------------------------------------------------------")
        json_request = json.loads(decoded_request_fixed, strict=False)
        return json_request
    except:
        return BadRequest()
        

def inference(text: str) -> str:
    """This function take a text as an input and
       predict whether or not it is a header. The input
       has to be line extracted by tesseract"""
    inputs = tokenizer(text, return_tensors="pt")

    inputs = {k:v.to(device) for k,v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1).tolist()[0]

    max_vale = max(predictions)
    idx = predictions.index(max_vale)
    return model.config.id2label[idx]
    
    
def get_text(request_body, request_content_type):
    # Get the text sent by Postman
    if request_content_type =="application/json":
        decoded_payload = _get_json(request_body)
        text = decoded_payload["query"]
        return text
        
    else:
        return BadContentType(["application/json"])
        
       
def handler(request_body, request_content_type):
    text = get_text(request_body, request_content_type)
    emotion_detected = inference(text)
    return {
        'headers': {'Content-Type' : 'application/json'},
        'statusCode': 200,
        'body': json.dumps({"message": "Lambda Container image invoked!",
                            "event": emotion_detected})
    }

