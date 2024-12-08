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
import logging
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_model_and_tokenizer():
    """Load model with proper error handling"""
    try:
        logger.info("Loading model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("AcuteShrewdSecurity/Llama-Phishsense-1B")
        base_model = AutoModelForCausalLM.from_pretrained("AcuteShrewdSecurity/Llama-Phishsense-1B")
        model = PeftModel.from_pretrained(base_model, "AcuteShrewdSecurity/Llama-Phishsense-1B")
        
        if torch.cuda.is_available():
            model = model.to('cuda')
            logger.info("Model loaded on GPU")
        else:
            logger.info("Model loaded on CPU")
            
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def load_emails(file_path):
    """Load and validate emails from JSONL"""
    try:
        emails = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('//'):
                    continue
                email = json.loads(line)
                if 'content' in email and 'email_type' in email:
                    emails.append(email)
        logger.info(f"Loaded {len(emails)} valid emails from {file_path}")
        return emails
    except Exception as e:
        logger.error(f"Error loading emails from {file_path}: {str(e)}")
        raise

def predict_single_email(model, tokenizer, text):
    """Make prediction for a single email"""
    prompt = f"""
You are a specialized phishing detection model. Analyze the following email content and determine if it is a phishing attempt.

TASK:
Classify the email content as either phishing (TRUE) or legitimate (FALSE).

EMAIL CONTENT: 
{text}

ANSWER (TRUE/FALSE):"""
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    
    if torch.cuda.is_available():
        inputs = {k: v.to('cuda') for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,  # More deterministic
            num_beams=3,     # Beam search
            early_stopping=True,
            top_p=1.0        # Set top_p to default value
        )
    
    pred_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Improved response parsing
    pred_text = pred_text.upper().strip()
    if pred_text.endswith('TRUE'):
        return 1, 0.9
    elif pred_text.endswith('FALSE'):
        return 0, 0.1
    return 0, 0.5  # Default case

def evaluate_model(model, tokenizer, texts):
    """Evaluate model with progress bar"""
    predictions = []
    predicted_probs = []
    
    for text in tqdm(texts, desc="Processing emails"):
        try:
            pred, prob = predict_single_email(model, tokenizer, text)
            predictions.append(pred)
            predicted_probs.append(prob)
        except Exception as e:
            logger.error(f"Error processing email: {str(e)}")
            predictions.append(0)
            predicted_probs.append(0.5)
    
    return predictions, predicted_probs

def main():
    try:
        # Load model and data
        model, tokenizer = load_model_and_tokenizer()
        
        valid_mails = load_emails('../examples/valid_mails.jsonl')
        phishing_mails = load_emails('../examples/phishing_mails.jsonl')
        emails = valid_mails + phishing_mails
        texts = [email['content'] for email in emails]
        true_labels = [1 if email['email_type'] == 'phishing' else 0 for email in emails]
        
        # Model evaluation
        predicted_labels, predicted_probs = evaluate_model(model, tokenizer, texts)
        
        # Create results directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f'results_{timestamp}'
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(true_labels, predicted_labels),
            'precision': precision_score(true_labels, predicted_labels),
            'recall': recall_score(true_labels, predicted_labels),
            'f1': f1_score(true_labels, predicted_labels),
            'roc_auc': roc_auc_score(true_labels, predicted_probs)
        }
        
        conf_matrix = confusion_matrix(true_labels, predicted_labels)
        
        # Save results and generate visualizations
        results = {
            'metrics': metrics,
            'confusion_matrix': conf_matrix.tolist(),
            'timestamp': timestamp,
            'num_samples': len(emails)
        }
        
        with open(f'{output_dir}/results.json', 'w') as f:
            json.dump(results, f, indent=4)
            
        # Generate visualizations
        plot_confusion_matrix(conf_matrix, output_dir)
        plot_roc_curve(true_labels, predicted_probs, metrics['roc_auc'], output_dir)
        
        logger.info(f"Analysis complete. Results saved in {output_dir}")
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise

def plot_confusion_matrix(conf_matrix, output_dir):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Valid', 'Phishing'],
                yticklabels=['Valid', 'Phishing'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'{output_dir}/confusion_matrix.png')
    plt.close()

def plot_roc_curve(true_labels, predicted_probs, roc_auc, output_dir):
    """Plot and save ROC curve"""
    fpr, tpr, _ = roc_curve(true_labels, predicted_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(f'{output_dir}/roc_curve.png')
    plt.close()

if __name__ == '__main__':
    main()