from flask import Flask, render_template, request
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel

app = Flask(__name__)

def get_confidence_level(score):
    if score >= 0.95:
        return "Very High"
    elif score >= 0.85:
        return "High"
    elif score >= 0.70:
        return "Moderate"
    return "Low"

def get_safety_explanation(is_safe, confidence):
    if is_safe:
        return f"SAFE: This content appears legitimate ({confidence} confidence)"
    return f"WARNING: Potential phishing attempt detected ({confidence} confidence)"

def normalize_result(label, score, raw_prediction):
    is_safe = label in ['benign', 'no phish', 'FALSE']
    normalized_label = 'Safe' if is_safe else 'Phishing'
    score_value = float(str(score).strip('%')) / 100 if isinstance(score, str) else score
    
    return {
        'is_safe': is_safe,
        'normalized_label': normalized_label,
        'score': f"{score_value:.2%}" if score_value != 'N/A' else 'N/A',
        'confidence_level': get_confidence_level(score_value) if score_value != 'N/A' else 'N/A',
        'safety_explanation': get_safety_explanation(is_safe, get_confidence_level(score_value) if score_value != 'N/A' else 'N/A'),
        'raw': raw_prediction
    }

# Model initialization
classifier = pipeline("text-classification", model="ealvaradob/bert-finetuned-phishing")
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
        try:
            text = request.form['text']
            
            # BERT Analysis
            bert_prediction = classifier(text)[0]
            bert_result = normalize_result(
                bert_prediction['label'],
                bert_prediction['score'],
                bert_prediction
            )
            
            # Llama Analysis
            prompt = f"Classify the following text as phishing or not. Respond with 'TRUE' or 'FALSE':\n\n{text}\nAnswer:"
            inputs = tokenizer(prompt, return_tensors="pt")
            
            if torch.cuda.is_available():
                inputs = {key: value.to('cuda') for key, value in inputs.items()}
            
            with torch.no_grad():
                output = llama_model.generate(**inputs, max_new_tokens=5, temperature=0.01, do_sample=False)
            
            llama_prediction = tokenizer.decode(output[0], skip_special_tokens=True).split("Answer:")[1].strip()
            # Estimate confidence based on temperature and sampling settings
            llama_confidence = 0.85 if llama_prediction in ['TRUE', 'FALSE'] else 0.5
            
            llama_result = normalize_result(
                llama_prediction,
                llama_confidence,
                llama_prediction
            )

        except Exception as e:
            return render_template('index.html', error=str(e))

    return render_template('index.html', bert_result=bert_result, llama_result=llama_result)

if __name__ == '__main__':
    app.run(debug=True, port=5002)