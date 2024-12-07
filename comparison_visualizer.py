import json
import matplotlib.pyplot as plt
import numpy as np
import os

# working direcotry
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

with open('llama_phish_demo/results_20241207_212417/results.json') as f:
    llama_results = json.load(f)

with open('bert-finetuned-phishing/results_20241207_212516/results.json') as f:
    bert_results = json.load(f)

# extract metrics to compare
metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
llama_values = [llama_results['metrics'][m] for m in metrics]
bert_values = [bert_results['metrics'][m] for m in metrics]

# comparison plot
plt.figure(figsize=(12, 6))
x = np.arange(len(metrics))
width = 0.35

plt.bar(x - width/2, llama_values, width, label='Llama-Phishsense-1B', color='skyblue')
plt.bar(x + width/2, bert_values, width, label='BERT-finetuned', color='lightgreen')

plt.ylabel('Score')
plt.title('Model Performance Comparison')
plt.xticks(x, metrics)
plt.legend()

# value labels (on top of bars)
for i, v in enumerate(llama_values):
    plt.text(i - width/2, v + 0.01, f'{v:.2f}', ha='center')
for i, v in enumerate(bert_values):
    plt.text(i + width/2, v + 0.01, f'{v:.2f}', ha='center')

plt.ylim(0, 1.1) 
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.savefig('model_comparison.png', bbox_inches='tight', dpi=300)
plt.show()  
plt.close()

# create confusion matrix
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# plot llama confusion matrix
llama_cm = [[llama_results['confusion_matrix']['true_negatives'], llama_results['confusion_matrix']['false_positives']],
            [llama_results['confusion_matrix']['false_negatives'], llama_results['confusion_matrix']['true_positives']]]
ax1.imshow(llama_cm, cmap='Blues')
ax1.set_title('Llama-Phishsense-1B\nConfusion Matrix')
ax1.set_xticks([0, 1])
ax1.set_yticks([0, 1])
ax1.set_xticklabels(['Negative', 'Positive'])
ax1.set_yticklabels(['Negative', 'Positive'])

# add values to llama matrix
for i in range(2):
    for j in range(2):
        ax1.text(j, i, llama_cm[i][j], ha='center', va='center')

# plot bert confusion matrix
bert_cm = [[bert_results['confusion_matrix']['true_negatives'], bert_results['confusion_matrix']['false_positives']],
           [bert_results['confusion_matrix']['false_negatives'], bert_results['confusion_matrix']['true_positives']]]
ax2.imshow(bert_cm, cmap='Greens')
ax2.set_title('BERT-finetuned\nConfusion Matrix')
ax2.set_xticks([0, 1])
ax2.set_yticks([0, 1])
ax2.set_xticklabels(['Negative', 'Positive'])
ax2.set_yticklabels(['Negative', 'Positive'])

# add values to bert matrix
for i in range(2):
    for j in range(2):
        ax2.text(j, i, bert_cm[i][j], ha='center', va='center')

plt.tight_layout()
plt.savefig('confusion_matrices_comparison.png', bbox_inches='tight', dpi=300)
plt.show() 
plt.close()

print("Visualizations have been created and saved as:")
print("1. model_comparison.png")
print("2. confusion_matrices_comparison.png")