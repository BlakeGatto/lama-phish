simple flask app I use test phishing detection in emails using two pre-trained models. Full credits go to the authors of the models used in this educational example.

- Utilizes two models for phishing detection:
  - **Model 1:** Llama-Phishsense-1B (by AcuteShrewdSecurity)
  - **Model 2:** bert-finetuned-phishing (by E. Alvarado)

## Credits
- **Model 1:** [Llama-Phishsense-1B](https://huggingface.co/AcuteShrewdSecurity/Llama-Phishsense-1B)  
  Author: AcuteShrewdSecurity

- **Model 2:** [bert-finetuned-phishing](https://huggingface.co/ealvaradob/bert-finetuned-phishing?library=transformers)  
  Author: E. Alvarado

## Info
  full_analysis_with_*_model.py files = full analysis with jsonl data in /examples/*.jsonl
  webgui.py = user input possible
  examples folder = json data for full_analysis
  comparison_visualizer.py = graphical comparison for both model results