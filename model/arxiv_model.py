"""
Inspired by HuggingFace Tutorial
Code to generate text summarizer, trained on arxiv papers.
Starting with importing necessary libraries and tools.
"""

pip install transformers
from transformers import pipeline
from datasets import load_dataset
from transformers import AutoTokenizer
from datasets import Dataset
from huggingface_hub import notebook_login
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

import accelerate
import os
import numpy as np
import pandas as pd
import pyarrow as pa
import evaluate

pip install kaggle rouge_score datasets

os.environ['KAGGLE_CONFIG_DIR'] = "/content"

#import dataset from Kaggle
!kaggle datasets download -d Cornell-University/arxiv
!chmod 600 /content/kaggle.json
!unzip /content/arxiv.zip -d arxiv-dataset

dataset = load_dataset("arxiv_dataset", data_dir='./arxiv-dataset/', split='train', ignore_verifications=True)

#preprocesing dataset
dataset = dataset.shuffle(seed=42)
dataset = dataset.select(range(25000))

df = pd.DataFrame(dataset)

#only keeping columns that are required
df = df[['abstract', 'title']]
df = df.rename(columns={"abstract": "text", "title": "summary"})
df = df.replace(r'\n',' ', regex=True)
pd.options.display.max_colwidth = 100

cutoff_summary = 5
cutoff_text = 20
df = df[(df['summary'].apply(lambda x: len(x.split()) >= cutoff_summary)) & (df['text'].apply(lambda x: len(x.split()) >= cutoff_text))]
df = df.sample(1000, random_state=43)

#split the dataset into train, val, and test
df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=44), [int(0.8*len(df)), int((0.9)*len(df))])

"""Data now split -- needs to be tokenized"""
checkpoint = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

prefix = "summarize: "
def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["text"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

    labels = tokenizer(text_target=examples["summary"], max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

#convert original dataset to Huggingface dataset
ds_train = Dataset(pa.Table.from_pandas(df_train))
tokenized_arxiv_train = ds_train.map(preprocess_function, batched=True)

ds_test = Dataset(pa.Table.from_pandas(df_test))
tokenized_arxiv_test = ds_test.map(preprocess_function, batched=True)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)

"""HuggingFace Evaluation"""

#!pip install evaluate
rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}

"""Train the model with clean data"""

model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

!pip install accelerate -U
!pip install transformers[torch]

notebook_login()

training_args = Seq2SeqTrainingArguments(
    output_dir="arxiv_model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=2,
    predict_with_generate=True,
    push_to_hub=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_arxiv_train,
    eval_dataset=tokenized_arxiv_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

""" Pushing the trained model """
trainer.train()
trainer.push_to_hub()
