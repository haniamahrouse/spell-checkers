from transformers import BartTokenizer, BartForConditionalGeneration

class BartSpellChecker:
    def __init__(self):
        self.tokenizer = BartTokenizer.from_pretrained("./bart_model")
        self.model = BartForConditionalGeneration.from_pretrained("./bart_model")

    def correct_sentence(self, text):
        input_text = f"fix spelling: {text}"

        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            padding=True
    )

        outputs = self.model.generate(
            inputs["input_ids"],
            max_length=64,
            num_beams=5,
            early_stopping=True
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)