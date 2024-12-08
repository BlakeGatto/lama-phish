from pathlib import Path
from typing import List, Dict, Any, Union, Tuple
import logging
from dataclasses import dataclass
import torch
from transformers import pipeline
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, roc_auc_score, roc_curve)
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class Config:
    model_name: str = "ealvaradob/bert-finetuned-phishing"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 32
    valid_mails_path: Path = Path('../examples/valid_mails.jsonl')
    phishing_mails_path: Path = Path('../examples/phishing_mails.jsonl')

class PhishingDetector:
    def __init__(self, config: Config):
        self.config = config
        self.label_mapping = {'benign': 'valid', 'phishing': 'phishing'}
        try:
            self.classifier = pipeline(
                "text-classification", 
                model=config.model_name,
                device=config.device,
                batch_size=config.batch_size
            )
            logging.info(f"Model loaded successfully on {config.device}")
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            raise

    def load_emails(self) -> List[Dict[str, Any]]:
        """Load and combine valid and phishing emails."""
        emails = []
        for path in [self.config.valid_mails_path, self.config.phishing_mails_path]:
            try:
                with path.open('r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('//'):
                            try:
                                emails.append(json.loads(line))
                            except json.JSONDecodeError as e:
                                logging.warning(f"Skipping invalid line: {line}")
                                continue  # Skip invalid lines
                logging.info(f"Loaded {len(emails)} emails from {path}")
            except Exception as e:
                logging.error(f"Error loading emails from {path}: {e}")
                raise
        return emails

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Make predictions in batches."""
        return self.classifier(texts, truncation=True)

    def evaluate(self, emails: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Evaluate model performance and generate metrics."""
        texts = [email['content'] for email in emails]
        true_labels = []
        predicted_labels = []
        predicted_probs = []

        for i in range(0, len(texts), self.config.batch_size):
            batch_texts = texts[i:i + self.config.batch_size]
            predictions = self.predict_batch(batch_texts)
            
            for j, pred in enumerate(predictions):
                email = emails[i + j]
                predicted_label = self.label_mapping[pred['label']]
                true_label = email['email_type']
                
                true_labels.append(1 if true_label == 'phishing' else 0)
                predicted_labels.append(1 if predicted_label == 'phishing' else 0)
                predicted_probs.append(pred['score'] if predicted_label == 'phishing' 
                                    else 1 - pred['score'])

        metrics = self.calculate_metrics(true_labels, predicted_labels, predicted_probs)
        results = self.generate_results_dict(emails, true_labels, predicted_labels, 
                                          predicted_probs, metrics['conf_matrix'])
        return metrics, results

    def calculate_metrics(self, true_labels: List[int], predicted_labels: List[int], 
                         predicted_probs: List[float]) -> Dict[str, Any]:
        """Calculate all evaluation metrics."""
        return {
            "accuracy": accuracy_score(true_labels, predicted_labels),
            "precision": precision_score(true_labels, predicted_labels),
            "recall": recall_score(true_labels, predicted_labels),
            "f1": f1_score(true_labels, predicted_labels),
            "conf_matrix": confusion_matrix(true_labels, predicted_labels),
            "roc_auc": roc_auc_score(true_labels, predicted_probs),
            "true_labels": true_labels,
            "predicted_labels": predicted_labels,
            "predicted_probs": predicted_probs
        }

    def generate_results_dict(self, emails: List[Dict[str, Any]], true_labels: List[int],
                            predicted_labels: List[int], predicted_probs: List[float],
                            conf_matrix: np.ndarray) -> Dict[str, Any]:
        """Generate structured results dictionary."""
        return {
            "model_name": self.config.model_name,
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
            }
        }

class Visualizer:
    @staticmethod
    def save_visualizations(metrics: Dict[str, Any], output_dir: Path) -> None:
        """Generate and save all visualizations."""
        output_dir.mkdir(exist_ok=True)
        
        # Confusion Matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(metrics['conf_matrix'], annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Valid', 'Phishing'],
                   yticklabels=['Valid', 'Phishing'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(output_dir / 'confusion_matrix.png')
        plt.close()

        # ROC Curve
        fpr, tpr, _ = roc_curve(metrics['true_labels'], metrics['predicted_probs'])
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {metrics["roc_auc"]:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.savefig(output_dir / 'roc_curve.png')
        plt.close()

def main():
    config = Config()
    output_dir = Path(f'results_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    
    try:
        detector = PhishingDetector(config)
        emails = detector.load_emails()
        metrics, results = detector.evaluate(emails)
        
        # Save results and visualizations
        output_dir.mkdir(exist_ok=True)
        with open(output_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=4)
            
        Visualizer.save_visualizations(metrics, output_dir)

        # Generate and save report
        report = f"""
        Phishing Detection Model Evaluation Report
        ========================================
        Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

        Dataset Overview:
        ----------------
        Total emails analyzed: {len(emails)}
        Valid emails: {results['dataset']['valid_emails']}
        Phishing emails: {results['dataset']['phishing_emails']}

        Performance Metrics:
        ------------------
        Accuracy: {results['metrics']['accuracy']:.2%}
        Precision: {results['metrics']['precision']:.2%}
        Recall: {results['metrics']['recall']:.2%}
        F1-Score: {results['metrics']['f1_score']:.2%}
        ROC-AUC: {results['metrics']['roc_auc']:.2f}

        Confusion Matrix:
        ---------------
        True Negatives: {results['confusion_matrix']['true_negatives']}
        False Positives: {results['confusion_matrix']['false_positives']}
        False Negatives: {results['confusion_matrix']['false_negatives']}
        True Positives: {results['confusion_matrix']['true_positives']}
        """

        with open(output_dir / 'report.txt', 'w') as f:
            f.write(report)

        print(report)
        logging.info(f"Results saved to {output_dir}")

    except Exception as e:
        logging.error(f"Error during execution: {e}")
        raise

if __name__ == "__main__":
    main()