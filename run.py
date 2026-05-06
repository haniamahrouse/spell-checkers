import random
import numpy as np
from datasets import load_dataset
from bart_spell_checker import BartSpellChecker

# ✅ FIRST define helper functions

def corrupt_text(text):
    words = text.split()
    new_words = []

    for word in words:
        # 40% chance to corrupt EACH word
        if random.random() < 0.4 and len(word) > 3:

            op = random.choice(["delete", "swap", "replace"])

            if op == "delete":
                i = random.randint(0, len(word)-1)
                word = word[:i] + word[i+1:]

            elif op == "swap" and len(word) > 4:
                i = random.randint(0, len(word)-2)
                word = word[:i] + word[i+1] + word[i] + word[i+2:]

            elif op == "replace":
                i = random.randint(0, len(word)-1)
                word = (
                    word[:i]
                    + random.choice("abcdefghijklmnopqrstuvwxyz")
                    + word[i+1:]
                )

        new_words.append(word)

    return " ".join(new_words)


def preprocess(example):
    return {
        "input": corrupt_text(example["text"]),
        "target": example["text"]
    }


def compute_word_accuracy(model, dataset):
    total_words = 0
    correct_words = 0

    for example in dataset:
        prediction = model.correct_sentence(example["input"])

        pred_words = prediction.split()
        target_words = example["target"].split()

        for p, t in zip(pred_words, target_words):
            total_words += 1
            if p == t:
                correct_words += 1

    return correct_words / total_words if total_words > 0 else 0


# ✅ THEN run everything

model = BartSpellChecker()

print("Evaluating model...")

dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test[:1%]")
dataset = dataset.map(preprocess)

accuracy = compute_word_accuracy(model, dataset)

print(f"\nModel Word Accuracy: {accuracy:.2%}\n")

print("BART Spell Checker Ready (type 'exit' to quit)")

while True:
    text = input("Enter sentence: ")

    if text.lower() == "exit":
        break

    corrected = model.correct_sentence(text)
    print("Corrected:", corrected)
