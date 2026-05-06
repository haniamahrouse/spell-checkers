import random
import numpy as np
from datasets import load_dataset
from bart_spell_checker import BartSpellChecker
from nltk.corpus import words

word_set = set(words.words())
# ✅ FIRST define helper functions


def realistic_typo(word):
    if len(word) < 3:
        return word

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


def compute_word_accuracy(model, dataset):
    total_words = 0
    correct_words = 0

    for example in dataset:
        raw_input = example["input"].replace("fix spelling: ", "")
        prediction = model.correct_sentence(raw_input)

        pred_words = prediction.split()
        target_words = example["target"].split()

        for p, t in zip(pred_words, target_words):
            total_words += 1
            if p == t:
                correct_words += 1

    return correct_words / total_words if total_words > 0 else 0


def postprocess(predicted_word):
    if predicted_word in word_set:
        return predicted_word
    return predicted_word


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
