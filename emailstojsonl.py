import csv
import json

input_csv = './emails.csv'
output_jsonl = 'emails.jsonl'

with open(input_csv, 'r', encoding='utf-8') as csv_file, open(output_jsonl, 'w', encoding='utf-8') as jsonl_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
        email_type = 'phishing' if row['spam'] == '1' else 'benign'
        content = row['text']
        json_entry = {
            'email_type': email_type,
            'content': content
        }
        jsonl_file.write(json.dumps(json_entry) + '\n')
