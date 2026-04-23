from collections import Counter
import re

# Load dataset
def words(text):
    return re.findall(r'\w+', text.lower())

with open('big.txt') as f:  # dataset file
    WORDS = Counter(words(f.read()))

def P(word):
    "Probability of word"
    return WORDS[word] / sum(WORDS.values())

def correction(word):
    "Most probable correction"
    return max(candidates(word), key=P)

def candidates(word):
    return (known([word]) or 
            known(edits1(word)) or 
            known(edits2(word)) or 
            [word])

def known(words):
    return set(w for w in words if w in WORDS)

def edits1(word):
    letters = 'abcdefghijklmnopqrstuvwxyz'
    splits = [(word[:i], word[i:]) for i in range(len(word)+1)]
    deletes = [L + R[1:] for L, R in splits if R]
    inserts = [L + c + R for L, R in splits for c in letters]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    return set(deletes + inserts + replaces)

def edits2(word):
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))

# Test
print(correction("recieve"))