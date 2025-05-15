import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, TrainingArguments, Trainer

# Define dataset class
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(self.texts[idx], padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# Load pre-trained BERT model and tokenizer
MODEL_NAME = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
bert_model = BertModel.from_pretrained(MODEL_NAME)

# Define custom classifier
class CustomBERTSentimentClassifier(nn.Module):
    def __init__(self, bert, num_labels=5):
        super(CustomBERTSentimentClassifier, self).__init__()
        self.bert = bert
        self.fc = nn.Linear(bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # Use CLS token representation
        logits = self.fc(pooled_output)
        return logits

# Create model instance
model = CustomBERTSentimentClassifier(bert_model)

# Example training data
texts = ["I love this!", "This is bad.", "It's okay.", "Fantastic experience!", "Very disappointing."]
labels = [4, 0, 2, 4, 1]  # Sentiment labels

# Create dataset and dataloader
dataset = SentimentDataset(texts, labels, tokenizer)
train_dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./sentiment_model",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

# Use Trainer API
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# Train model
trainer.train()

# Save model
torch.save(model.state_dict(), "./fine_tuned_custom_bert.pth")
tokenizer.save_pretrained("./fine_tuned_custom_bert")
