from flask import Flask, render_template, request
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel

app = Flask(__name__)

# Modell 1: BERT
classifier = pipeline("text-classification", model="ealvaradob/bert-finetuned-phishing")

# Modell 2: Llama-Phishsense-1B
tokenizer = AutoTokenizer.from_pretrained("AcuteShrewdSecurity/Llama-Phishsense-1B")
base_model = AutoModelForCausalLM.from_pretrained("AcuteShrewdSecurity/Llama-Phishsense-1B")
llama_model = PeftModel.from_pretrained(base_model, "AcuteShrewdSecurity/Llama-Phishsense-1B")

if torch.cuda.is_available():
    llama_model = llama_model.to('cuda')

@app.route('/', methods=['GET', 'POST'])
def home():
    bert_result = None
    llama_result = None
    if request.method == 'POST':
        text = request.form['text']
        
        # Modell 1: BERT
        bert_prediction = classifier(text)[0]
        bert_result = {
            'label': bert_prediction['label'],
            'score': f"{bert_prediction['score']:.2%}"
        }
        
        # Modell 2: Llama-Phishsense-1B
        prompt = f"Classify the following text as phishing or not. Respond with 'TRUE' or 'FALSE':\n\n{text}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt")
        
        if torch.cuda.is_available():
            inputs = {key: value.to('cuda') for key, value in inputs.items()}
        
        with torch.no_grad():
            output = llama_model.generate(**inputs, max_new_tokens=5, temperature=0.01, do_sample=False)
        
        llama_prediction = tokenizer.decode(output[0], skip_special_tokens=True).split("Answer:")[1].strip()
        llama_result = {'prediction': llama_prediction}

    return render_template('index.html', bert_result=bert_result, llama_result=llama_result)

if __name__ == '__main__':
    app.run(debug=True, port=5002)
