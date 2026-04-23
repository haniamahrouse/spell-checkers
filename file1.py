import random
import re
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from symspellpy import SymSpell, Verbosity

nltk.download("punkt")

def simple_tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())
# -----------------------------
# 1. Load Wikipedia dataset
# -----------------------------
print("Loading Wikipedia dataset...")
from datasets import load_dataset

dataset = load_dataset(
    "wikimedia/wikipedia",
    "20231101.en",
    split="train[:1%]"
)

texts = [item["text"] for item in dataset]

# -----------------------------
# 2. Train/Test Split (70/30)
# -----------------------------
random.shuffle(texts)
split_idx = int(0.7 * len(texts))

train_texts = texts[:split_idx]
test_texts = texts[split_idx:]

print("Train size:", len(train_texts))
print("Test size:", len(test_texts))

# -----------------------------
# 3. Build vocabulary (for spell checking)
# -----------------------------
print("Building vocabulary...")

words = Counter()

for text in train_texts:
    tokens = simple_tokenize(text)
    words.update(tokens)

vocab = set(words.keys())

# -----------------------------
# 4. SymSpell for fast candidate generation
# -----------------------------
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)

for word, freq in words.items():
    sym_spell.create_dictionary_entry(word, freq)

# -----------------------------
# 5. Load BERT model (context understanding)
# -----------------------------
print("Loading BERT model...")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased")
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# -----------------------------
# 6. Utility: score sentence
# -----------------------------
def score_sentence(sentence):
    tokens = tokenizer(sentence, return_tensors="pt").to(device)
    with torch.inference_mode():
        output = model(**tokens, labels=tokens["input_ids"])
        loss = output.loss
    return -loss.item()  # higher is better


# -----------------------------
# 7. Generate correction candidates
# -----------------------------
def get_candidates(word):
    suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)

    candidates = [s.term for s in suggestions]

    candidates = [
        w for w in candidates
        if w.isalpha() and abs(len(w) - len(word)) <= 2
    ]

    return candidates[:3] if candidates else [word]
# -----------------------------
# 8. Spell correction function
# -----------------------------
def correct_sentence(sentence):
    tokens = sentence.split()
    mask_positions = []
    candidate_lists = []

    # Step 1
    for i, word in enumerate(tokens):
        word_lower = word.lower()

        if word_lower in vocab:
            continue

        if word[0].isupper():
            continue

        if not word.isalpha():
            continue

        candidates = get_candidates(word_lower)

        if len(candidates) > 1:
            mask_positions.append(i)
            candidate_lists.append(candidates)
            tokens[i] = tokenizer.mask_token

    if not mask_positions:
        return sentence

    # Step 2
    masked_sentence = " ".join(tokens)
    inputs = tokenizer(masked_sentence, return_tensors="pt").to(device)

    with torch.inference_mode():
        outputs = model(**inputs)

    logits = outputs.logits
    input_ids = inputs.input_ids[0]

    # Step 3 (THIS MUST BE INSIDE FUNCTION)
    import torch.nn.functional as F

    for pos, candidates in zip(mask_positions, candidate_lists):
        mask_index = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[0][0]

        probs = F.softmax(logits[0, mask_index], dim=-1)

        best_word = candidates[0]
        best_score = -1

        for cand in candidates:
            token_id = tokenizer.convert_tokens_to_ids(cand)

            if token_id == tokenizer.unk_token_id:
                continue

            score = probs[token_id].item() * (words[cand] + 1)

            if score > best_score:
                best_score = score
                best_word = cand

        # preserve capitalization
        original_word = sentence.split()[pos]
        if original_word.istitle():
            best_word = best_word.capitalize()

        tokens[pos] = best_word

        input_ids[mask_index] = 0

    return " ".join(tokens)
# -----------------------------
# 9. Evaluation (30% test set)
# -----------------------------
print("\nEvaluating on test set...\n")

for text in test_texts[:10]:
    sentences = re.split(r"[.!?]", text)

    for sent in sentences[:2]:
        if len(sent.strip()) < 5:
            continue

        # simulate typo (for testing)
        words_sent = sent.split()
        if len(words_sent) < 3:
            continue

        idx = random.randint(0, len(words_sent)-1)
        words_sent[idx] = words_sent[idx][::-1]  # fake typo
        noisy = " ".join(words_sent)

        corrected = correct_sentence(noisy)

        print("Original  :", sent)
        print("Noisy     :", noisy)
        print("Corrected :", corrected)
        print("-" * 50)


# -----------------------------
# 10. Interactive mode
# -----------------------------
print("\nSpell Checker Ready (type 'exit')\n")

while True:
    text = input("Enter text: ")
    if text.lower() == "exit":
        break

    print("Corrected:", correct_sentence(text))
    print("-" * 50)