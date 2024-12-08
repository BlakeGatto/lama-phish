import json
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import seaborn as sns
from typing import Dict, List, Union
from PIL import Image

def load_results(filepath: str) -> Dict:
    """Load and validate results from JSON file."""
    try:
        with open(filepath) as f:
            results = json.load(f)
        validate_results(results)
        return results
    except FileNotFoundError:
        raise FileNotFoundError(f"Results file not found: {filepath}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in file: {filepath}")

def validate_results(results: Dict) -> None:
    """Validate required fields in results."""
    required_fields = {'metrics', 'confusion_matrix'}
    if not all(field in results for field in required_fields):
        raise ValueError("Missing required fields in results")

def normalize_metrics(results: Dict) -> Dict:
    """Normalize metric names and ensure consistent format."""
    metrics = results.get('metrics', {})
    if 'f1' in metrics and 'f1_score' not in metrics:
        metrics['f1_score'] = metrics.pop('f1')
    return metrics

def get_confusion_matrix(results: Dict) -> List[List[int]]:
    """Extract confusion matrix in consistent format."""
    cm = results['confusion_matrix']
    if isinstance(cm, list):
        return cm
    return [
        [cm.get('true_negatives', 0), cm.get('false_positives', 0)],
        [cm.get('false_negatives', 0), cm.get('true_positives', 0)]
    ]

def create_performance_comparison(llama_metrics: Dict, bert_metrics: Dict, metrics: List[str]) -> None:
    """Create and save performance comparison plot."""
    plt.figure(figsize=(15, 8))
    x = np.arange(len(metrics))
    width = 0.35

    # Plot bars and labels
    llama_values = [llama_metrics.get(m, 0) for m in metrics]
    bert_values = [bert_metrics.get(m, 0) for m in metrics]
    
    plt.bar(x - width/2, llama_values, width, label='Llama-Phishsense-1B', color='skyblue')
    plt.bar(x + width/2, bert_values, width, label='BERT-finetuned', color='lightgreen')

    plt.ylabel('Score', fontsize=12)
    plt.title('Model Performance Comparison', fontsize=14, pad=20)
    plt.xticks(x, metrics, fontsize=11)
    plt.legend(fontsize=11, bbox_to_anchor=(1.05, 1), loc='upper left')

    # Adjusted value labels
    for i, v in enumerate(llama_values):
        plt.text(i - width/2, v + 0.02, f'{v:.2f}', ha='center', fontsize=10)
    for i, v in enumerate(bert_values):
        plt.text(i + width/2, v + 0.02, f'{v:.2f}', ha='center', fontsize=10)

    plt.ylim(0, 1.2)  # Increased y-limit for better label visibility
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout(pad=2.0)  # More padding

    plt.savefig('model_comparison.png', bbox_inches='tight', dpi=300)
    plt.show()  
    plt.close()

def combine_roc_curves(llama_path: str, bert_path: str) -> None:
    """Combine existing ROC curves from both models into one comparison image."""
    try:
        # Load the existing ROC curve images
        llama_roc = Image.open(os.path.join(llama_path, 'roc_curve.png'))
        bert_roc = Image.open(os.path.join(bert_path, 'roc_curve.png'))

        # Create a new figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        # Display the images
        ax1.imshow(llama_roc)
        ax1.axis('off')
        ax1.set_title('Llama-Phishsense-1B ROC Curve', pad=20)

        ax2.imshow(bert_roc)
        ax2.axis('off')
        ax2.set_title('BERT-finetuned ROC Curve', pad=20)

        plt.suptitle('ROC Curve Comparison', fontsize=16, y=1.05)
        plt.tight_layout()
        plt.savefig('roc_curves_comparison.png', bbox_inches='tight', dpi=300)
        plt.close()

    except Exception as e:
        print(f"Error combining ROC curves: {e}")

def create_confusion_matrix_comparison(llama_results: Dict, bert_results: Dict) -> None:
    """Create and save confusion matrix comparison visualization."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Get confusion matrices
    llama_cm = np.array(get_confusion_matrix(llama_results))
    bert_cm = np.array(get_confusion_matrix(bert_results))
    
    # Calculate percentages
    llama_total = llama_cm.sum()
    bert_total = bert_cm.sum()
    llama_perc = llama_cm / llama_total * 100
    bert_perc = bert_cm / bert_total * 100
    
    # Plot matrices
    labels = ['Legitim', 'Phishing']
    for ax, cm, perc, title in [(ax1, llama_cm, llama_perc, 'Llama-Phishsense-1B'),
                               (ax2, bert_cm, bert_perc, 'BERT-finetuned')]:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=labels, yticklabels=labels)
        
        # Add percentage annotations
        for i in range(2):
            for j in range(2):
                ax.text(j+0.5, i+0.7, f'({perc[i,j]:.1f}%)',
                       ha='center', va='center')
        
        ax.set_title(f'{title}\nConfusion Matrix', pad=20)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')

    # Add explanation text
    plt.figtext(0.15, -0.1,
                'Interpretationshinweise:\n' +
                '• True Negatives (oben links): Korrekt als legitim klassifiziert\n' +
                '• False Positives (oben rechts): Fälschlicherweise als Phishing klassifiziert\n' +
                '• False Negatives (unten links): Nicht erkanntes Phishing\n' +
                '• True Positives (unten rechts): Korrekt erkanntes Phishing\n' +
                'Die Prozentwerte zeigen den Anteil an der Gesamtmenge.',
                fontsize=10, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    plt.tight_layout()
    plt.savefig('confusion_matrices_comparison.png', bbox_inches='tight', dpi=300, pad_inches=0.5)
    plt.close()

def main():
    """Main execution function."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    try:
        llama_path = 'llama_phish_demo/results_realdaten'
        bert_path = 'bert-finetuned-phishing/results_realdaten'
        
        llama_results = load_results(f'{llama_path}/results.json')
        bert_results = load_results(f'{bert_path}/results.json')

        llama_metrics = normalize_metrics(llama_results)
        bert_metrics = normalize_metrics(bert_results)

        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        create_performance_comparison(llama_metrics, bert_metrics, metrics)
        combine_roc_curves(llama_path, bert_path)
        create_confusion_matrix_comparison(llama_results, bert_results) 

    except Exception as e:
        print(f"Error during visualization: {e}")
        raise

if __name__ == "__main__":
    main()
