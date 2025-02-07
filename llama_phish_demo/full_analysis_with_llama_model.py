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
from huggingface_hub import login  # Import the login function
import math

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def login_hugging_face():
    """Login to Hugging Face using a token from environment or prompt the user."""
    try:
        token = # Insert Token Here
        if token is None:
            token = input("Please enter your Hugging Face token: ")
        login(token=token)
        logger.info("Successfully logged in to Hugging Face")
    except Exception as e:
        logger.error(f"Failed to login to Hugging Face: {str(e)}")
        raise

def load_model_and_tokenizer():
    """Load model with proper error handling"""
    try:
        logger.info("Loading model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("AcuteShrewdSecurity/Llama-PhishSense-1B")
        base_model = AutoModelForCausalLM.from_pretrained("AcuteShrewdSecurity/Llama-PhishSense-1B")
        model = PeftModel.from_pretrained(base_model, "AcuteShrewdSecurity/Llama-PhishSense-1B")
        
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

def compute_candidate_logprob(model, tokenizer, prompt, candidate):
    """
    Compute the log-likelihood of a candidate answer given the prompt.
    This is done by concatenating the prompt with the candidate and computing
    the sum of log probabilities for the candidate tokens.
    """
    # Concatenate prompt and candidate text
    full_text = prompt + candidate
    # Tokenize full text
    full_inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512)
    if torch.cuda.is_available():
        full_inputs = {k: v.to('cuda') for k, v in full_inputs.items()}
    with torch.no_grad():
        outputs = model(**full_inputs)
    logits = outputs.logits  # shape: (1, seq_length, vocab_size)

    # Tokenize candidate separately (do not add special tokens)
    candidate_ids = tokenizer(candidate, add_special_tokens=False, return_tensors="pt").input_ids[0]
    # Tokenize prompt to determine where candidate tokens begin
    prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).input_ids[0]
    prompt_length = prompt_ids.shape[0]
    
    total_logprob = 0.0
    # For each token in the candidate, the probability is computed from the logits at position (prompt_length - 1 + i)
    # because the model predicts token i using the context up to token i-1.
    for i, token_id in enumerate(candidate_ids):
        pos = prompt_length - 1 + i
        token_logits = logits[0, pos, :]
        log_probs = torch.log_softmax(token_logits, dim=-1)
        total_logprob += log_probs[token_id].item()
    return total_logprob

def predict_single_email(model, tokenizer, text):
    """Make prediction for a single email by extracting probabilities from a causal LM"""
    prompt = f"""
Task: Phishing Detection Classification

Instructions:

* Read the text provided below.
* Determine whether the text is a phishing attempt. Consider whether it contains indicators such as urgent requests for sensitive information, suspicious links, impersonation of trusted sources, or other common phishing characteristics.
* Respond with "TRUE" if the text is identified as a phishing attempt, or "FALSE" if it is not.
Text:
{text}
Answer:

"""
    # Compute log-likelihood for each candidate
    logp_true = compute_candidate_logprob(model, tokenizer, prompt, "TRUE")
    logp_false = compute_candidate_logprob(model, tokenizer, prompt, "FALSE")
    
    # Convert log-likelihoods to probabilities via softmax
    p_true = math.exp(logp_true)
    p_false = math.exp(logp_false)
    total = p_true + p_false
    prob_true = p_true / total
    threshold_log_odds = math.log(0.999 / (1 - 0.999))
    
    predicted_label = 1 if (logp_true - logp_false) >= threshold_log_odds else 0
    
    # Optionally, print the computed probabilities for debugging
    print(f"Computed probabilities -> TRUE: {prob_true:.4f}, FALSE: {1-prob_true:.4f}")
    
    return predicted_label, prob_true

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

def main():
    try:
        # Login to Hugging Face
        login_hugging_face()
        
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

if __name__ == '__main__':
    main()
