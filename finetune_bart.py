import random
import string
from datasets import load_dataset
from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments


dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:5%]")

def realistic_typo(word):
    if len(word) < 3:
        return word

    import random

    ops = ["delete", "swap", "replace", "insert"]
    op = random.choice(ops)

    i = random.randint(0, len(word) - 2)

    if op == "delete":
        return word[:i] + word[i+1:]

    elif op == "swap":
        return word[:i] + word[i+1] + word[i] + word[i+2:]

    elif op == "replace":
        return word[:i] + random.choice("abcdefghijklmnopqrstuvwxyz") + word[i+1:]

    elif op == "insert":
        return word[:i] + random.choice("abcdefghijklmnopqrstuvwxyz") + word[i:]

    return word

def corrupt_text(text):
    words = text.split()

    if len(words) == 0:
        return text

    i = random.randint(0, len(words)-1)
    words[i] = realistic_typo(words[i])

    return " ".join(words)


def preprocess(example):
    corrupted = corrupt_text(example["text"])
    return {
        "input": f"fix spelling: {corrupted}",
        "target": example["text"]
    }
dataset = dataset.map(preprocess)

# Train/test split (70/30)
dataset = dataset.train_test_split(test_size=0.3)

train_dataset = dataset["train"]
test_dataset = dataset["test"]


tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")


def tokenize(example):
    inputs = tokenizer(
        example["input"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

    labels = tokenizer(
        example["target"],
        truncation=True,
        padding="max_length",
        max_length=64
    )["input_ids"]

    # ignore padding in loss
    labels = [
        -100 if token == tokenizer.pad_token_id else token
        for token in labels
    ]

    inputs["labels"] = labels
    return inputs

train_dataset = train_dataset.map(tokenize)
test_dataset = test_dataset.map(tokenize)


training_args = TrainingArguments(
    output_dir="./bart_model",
    num_train_epochs=5,
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    logging_steps=100,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer
)

trainer.train()

# Save model
model.save_pretrained("./bart_model")
tokenizer.save_pretrained("./bart_model")