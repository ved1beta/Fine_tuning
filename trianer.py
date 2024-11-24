
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
dataset = load_dataset("stanfordnlp/imdb" , split  = "train")
from datasets import load_dataset
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import numpy as np
from datasets import Dataset



dataset = load_dataset("yelp_review_full")
trian_dataset = dataset["train"].to_pandas()
train_df , test_df = train_test_split(
    trian_dataset ,
    test_size=0.2,
    random_state=99
)
trian_dataset= Dataset.from_pandas(train_df)

test_dataset= Dataset.from_pandas(test_df)


from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)
tokenized_datasets = dataset.map(tokenize_function, batched=True)


small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
tokenized_train = trian_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)


from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels = 5)

import numpy as np 
import evaluate

matric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return matric.compute(predictions=predictions, references=labels)


from transformers import TrainingArguments , Trainer
training_args = TrainingArguments(output_dir="test-trianer", eval_strategy="epoch")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)
trainer.train()