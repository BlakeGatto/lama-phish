from flask import Flask, render_template, request
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

app = Flask(__name__)

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained("AcuteShrewdSecurity/Llama-Phishsense-1B")
base_model = AutoModelForCausalLM.from_pretrained("AcuteShrewdSecurity/Llama-Phishsense-1B")
model = PeftModel.from_pretrained(base_model, "AcuteShrewdSecurity/Llama-Phishsense-1B")

if torch.cuda.is_available():
    model = model.to('cuda')

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    if request.method == 'POST':
        text = request.form['text']
        prompt = f"Classify the following text as phishing or not. Respond with 'TRUE' or 'FALSE':\n\n{text}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt")
        
        if torch.cuda.is_available():
            inputs = {key: value.to('cuda') for key, value in inputs.items()}

        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=5, temperature=0.01, do_sample=False)
        
        prediction = tokenizer.decode(output[0], skip_special_tokens=True).split("Answer:")[1].strip()
        result = {'prediction': prediction}
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True, port=5001)