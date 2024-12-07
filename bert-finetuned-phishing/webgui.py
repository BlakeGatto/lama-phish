from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)

classifier = pipeline("text-classification", model="ealvaradob/bert-finetuned-phishing")

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    if request.method == 'POST':
        text = request.form['text']
        prediction = classifier(text)[0]
        result = {
            'label': prediction['label'],
            'score': f"{prediction['score']:.2%}"
        }
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)