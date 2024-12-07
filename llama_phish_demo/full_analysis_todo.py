import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
from datetime import datetime
import os

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained("AcuteShrewdSecurity/Llama-Phishsense-1B")
base_model = AutoModelForCausalLM.from_pretrained("AcuteShrewdSecurity/Llama-Phishsense-1B")
model = PeftModel.from_pretrained(base_model, "AcuteShrewdSecurity/Llama-Phishsense-1B")

if torch.cuda.is_available():
    model = model.to('cuda')

def load_emails(file_path):
    emails = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('//'):
                continue
            emails.append(json.loads(line))
    return emails

def evaluate_model(texts, true_labels):
    predictions = []
    predicted_probs = []
    
    for i, text in enumerate(texts):
        try:
            print(f"\rProcessing email {i+1}/{len(texts)}", end="")
            prompt = f"""
{text}

Is this a phishing email? Answer only with TRUE or FALSE:"""
            
            inputs = tokenizer(prompt, return_tensors="pt")
            
            if torch.cuda.is_available():
                inputs = {key: value.to('cuda') for key, value in inputs.items()}

            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=5,
                    do_sample=True,     
                    temperature=0.7,     
                    top_p=0.9,          
                    num_beams=1,      
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            pred_text = tokenizer.decode(output[0], skip_special_tokens=True)
            # Look for the answer at the end of the response
            if 'TRUE' in pred_text.upper().split()[-1:]:
                pred = 1
                prob = 0.9  # High confidence for positive prediction
            else:
                pred = 0
                prob = 0.1  # Low confidence for negative prediction
            
            predictions.append(pred)
            predicted_probs.append(prob)
            
        except KeyboardInterrupt:
            print("\nEvaluation interrupted by user. Processing results so far...")
            break
        except Exception as e:
            print(f"\nError processing email {i+1}: {str(e)}")
            predictions.append(0)
            predicted_probs.append(0.5)  # 0.5 for errors to indicate uncertainty
    
    print("\nEvaluation complete.")
    return predictions, predicted_probs

def main():
    try:
        # Load data from JSONL files
        valid_mails_path = '../examples/valid_mails.jsonl'
        phishing_mails_path = '../examples/phishing_mails.jsonl'
        
        print("Loading emails...")
        emails = load_emails(valid_mails_path) + load_emails(phishing_mails_path)
        texts = [email['content'] for email in emails]
        true_labels = [1 if email['email_type'] == 'phishing' else 0 for email in emails]
        
        print("Evaluating model...")
        predicted_labels, predicted_probs = evaluate_model(texts, true_labels)
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels)
        recall = recall_score(true_labels, predicted_labels)
        f1 = f1_score(true_labels, predicted_labels)
        conf_matrix = confusion_matrix(true_labels, predicted_labels)
        roc_auc = roc_auc_score(true_labels, predicted_probs)
        
        # Create output directory
        output_dir = f'results_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        os.makedirs(output_dir, exist_ok=True)
        
        # Visualize Confusion Matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Valid', 'Phishing'],
                    yticklabels=['Valid', 'Phishing'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'{output_dir}/confusion_matrix.png')
        plt.close()

        # Plot ROC Curve
        fpr, tpr, _ = roc_curve(true_labels, predicted_probs)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.savefig(f'{output_dir}/roc_curve.png')
        plt.close()

        # Generate and save report
        report = f"""
        Phishing Detection Model Evaluation Report
        ========================================
        Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

        Dataset Overview:
        ----------------
        Total emails analyzed: {len(emails)}
        Valid emails: {sum(1 for label in true_labels if label == 0)}
        Phishing emails: {sum(1 for label in true_labels if label == 1)}

        Performance Metrics:
        ------------------
        Accuracy: {accuracy:.2%}
        Precision: {precision:.2%}
        Recall: {recall:.2%}
        F1-Score: {f1:.2%}
        ROC-AUC: {roc_auc:.2f}

        Confusion Matrix:
        ---------------
        True Negatives: {conf_matrix[0][0]}
        False Positives: {conf_matrix[0][1]}
        False Negatives: {conf_matrix[1][0]}
        True Positives: {conf_matrix[1][1]}
        """

        with open(f'{output_dir}/report.txt', 'w') as f:
            f.write(report)

        print(report)
        print(f"\nResults saved in: {output_dir}")
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting...")
        return
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        return

if __name__ == '__main__':
    main()