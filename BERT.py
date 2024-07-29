import pandas as pd

import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

data = pd.read_csv('Merged_dataset.csv')

# Initialize the tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=4)

# Preprocess the data
def preprocess_data(posts, annotations):
    inputs = tokenizer(posts, padding=True, truncation=True, return_tensors="pt")
    labels = torch.tensor(annotations).float()
    return inputs, labels

# Convert annotations to list of integers
data['annotate'] = data['annotate'].apply(lambda x: eval(x))

# Split the data
train_texts, val_texts, train_labels, val_labels = train_test_split(data['post'].tolist(), data['annotate'].tolist(), test_size=0.2)

train_inputs, train_labels = preprocess_data(train_texts, train_labels)
val_inputs, val_labels = preprocess_data(val_texts, val_labels)

# Define a custom dataset class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

# Create datasets
train_dataset = CustomDataset(train_inputs, train_labels)
val_dataset = CustomDataset(val_inputs, val_labels)


# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,
    per_device_train_batch_size=20,
    per_device_eval_batch_size=20,
    warmup_steps=600,
    #learning_rate=1e-3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10
)

# Define a compute metrics function
def compute_metrics(pred):
    labels = pred.label_ids
    preds = torch.sigmoid(torch.tensor(pred.predictions)).round()
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()

# Print evaluation results
print(f"Evaluation results: {eval_results}")




# Check if GPU is available and use it if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move the model to the appropriate device
model.to(device)

# Define the prediction function
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    outputs = model(**inputs)
    predictions = torch.sigmoid(outputs.logits).detach().cpu().numpy()
    return predictions


# Test the prediction function
sample_text = "will this work out?"
prediction = predict(sample_text)
print(f"Prediction for '{sample_text}': {prediction}")
