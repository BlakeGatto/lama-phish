from transformers import pipeline
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
from datetime import datetime

classifier = pipeline("text-classification", model="ealvaradob/bert-finetuned-phishing")

# paths to the valid and phishing emails JSONL files
valid_mails_path = '../examples/valid_mails.jsonl'
phishing_mails_path = '../examples/phishing_mails.jsonl'

def load_emails(file_path):
    emails = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('//'):
                continue
            emails.append(json.loads(line))
    return emails

# load mails and combine "/examples/valid_mails.jsonl" and "/examples/phishing_mails.jsonl"
emails = load_emails(valid_mails_path) + load_emails(phishing_mails_path)

# classify the emails with the model and collect the results
correct_predictions = 0
total_emails = len(emails)

# var decalration to store the true labels and predictions
true_labels = []
predicted_labels = []
predicted_probs = []

# map the model labels to 'valid' and 'phishing' from the json files
label_mapping = {'benign': 'valid', 'phishing': 'phishing'}

for email in emails:
    text = email['content']
    prediction = classifier(text)[0]
    predicted_label = label_mapping[prediction['label']]
    true_label = email['email_type']
    # labels to binary values: 'phishing' = 1, 'valid' = 0
    true_labels.append(1 if true_label == 'phishing' else 0)
    predicted_labels.append(1 if predicted_label == 'phishing' else 0)
    predicted_probs.append(prediction['score'] if predicted_label == 'phishing' else 1 - prediction['score'])
    if predicted_label == true_label:
        correct_predictions += 1

# use sklearn to calculate the metrics such as 
# accuracy, precision, recall, f1 score, confusion matrix and roc auc score
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels)
recall = recall_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels)
conf_matrix = confusion_matrix(true_labels, predicted_labels)
roc_auc = roc_auc_score(true_labels, predicted_probs)

def generate_results_dict(emails, true_labels, predicted_labels, predicted_probs, conf_matrix, output_dir):
    results = {
        "model_name": "bert-finetuned-phishing",
        "timestamp": datetime.now().isoformat(),
        "dataset": {
            "total_emails": len(emails),
            "valid_emails": sum(1 for label in true_labels if label == 0),
            "phishing_emails": sum(1 for label in true_labels if label == 1)
        },
        "metrics": {
            "accuracy": float(accuracy_score(true_labels, predicted_labels)),
            "precision": float(precision_score(true_labels, predicted_labels)),
            "recall": float(recall_score(true_labels, predicted_labels)),
            "f1_score": float(f1_score(true_labels, predicted_labels)),
            "roc_auc": float(roc_auc_score(true_labels, predicted_probs))
        },
        "confusion_matrix": {
            "true_negatives": int(conf_matrix[0][0]),
            "false_positives": int(conf_matrix[0][1]),
            "false_negatives": int(conf_matrix[1][0]),
            "true_positives": int(conf_matrix[1][1])
        },
        "output_files": {
            "confusion_matrix_plot": f"{output_dir}/confusion_matrix.png",
            "roc_curve_plot": f"{output_dir}/roc_curve.png",
            "report_file": f"{output_dir}/report.txt"
        }
    }
    return results

# Create output directory for results
output_dir = f'results_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
import os
os.makedirs(output_dir, exist_ok=True)

# use the calculated metrics to display the accuracy
# such as the total emails, correct predictions and accuracy
accuracy = correct_predictions / total_emails * 100
print(f"Total emails: {total_emails}")
print(f"Correct predictions: {correct_predictions}")
print(f"Accuracy: {accuracy:.2f}%")

# print the results
print(f"Accuracy: {accuracy:.2%}")
print(f"Precision: {precision:.2%}")
print(f"Recall: {recall:.2%}")
print(f"F1-Score: {f1:.2%}")
print("Confusion Matrix:")
print(conf_matrix)
print(f"ROC-AUC: {roc_auc:.2f}")

# After calculating metrics and before visualization
results = generate_results_dict(emails, true_labels, predicted_labels, predicted_probs, conf_matrix, output_dir)

# Save results as JSON
with open(f'{output_dir}/results.json', 'w') as f:
    json.dump(results, f, indent=4)

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

# Generate detailed report
report = f"""
Phishing Detection Model Evaluation Report
========================================
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Dataset Overview:
----------------
Total emails analyzed: {total_emails}
Correct predictions: {correct_predictions}

Performance Metrics:
------------------
Accuracy: {accuracy:.2f}% 
-> This means the model correctly classified {accuracy:.2f}% of all emails.

Precision: {precision:.2%}
-> Out of all emails flagged as phishing, {precision:.2%} were actually phishing.

Recall: {recall:.2%}
-> The model successfully identified {recall:.2%} of all actual phishing emails.

F1-Score: {f1:.2%}
-> This is the harmonic mean of precision and recall, indicating overall performance.

ROC-AUC: {roc_auc:.2f}
-> A score close to 1.0 indicates excellent classification performance.

Confusion Matrix Interpretation:
-----------------------------
True Negatives (Valid emails correctly identified): {conf_matrix[0][0]}
False Positives (Valid emails incorrectly flagged as phishing): {conf_matrix[0][1]}
False Negatives (Phishing emails missed): {conf_matrix[1][0]}
True Positives (Phishing emails correctly identified): {conf_matrix[1][1]}

Key Insights:
-----------
1. False Positive Rate: {conf_matrix[0][1]/(conf_matrix[0][0] + conf_matrix[0][1]):.2%}
   -> This represents legitimate emails incorrectly marked as phishing
2. False Negative Rate: {conf_matrix[1][0]/(conf_matrix[1][0] + conf_matrix[1][1]):.2%}
   -> This represents phishing emails that went undetected

Visualizations saved:
-------------------
- Confusion Matrix: {output_dir}/confusion_matrix.png
- ROC Curve: {output_dir}/roc_curve.png
"""

# Save report
with open(f'{output_dir}/report.txt', 'w') as f:
    f.write(report)

# Print report to console
print(report)