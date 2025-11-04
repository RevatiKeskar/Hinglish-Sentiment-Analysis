
import os
os.environ["USE_TF"] = "0"

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from flask import Flask, render_template, request

app = Flask(__name__)

# Load model and tokenizer
model_path = "model"  # folder name where model is placed
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)

# Define labels
labels = ["Negative", "Neutral", "Positive"]

@app.route("/", methods=["GET", "POST"])
def home():
    result = ""
    if request.method == "POST":
        text = request.form["user_input"]
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=1).item()
            result = labels[prediction]
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
