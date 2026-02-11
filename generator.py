from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from config import (
    GENERATION_MODEL_NAME,
    MAX_NEW_TOKENS,
    TEMPERATURE,
)


class SanskritGenerator:
    """
    Simple wrapper around a multilingual seq2seq model (mT5-small) for CPU text generation.
    """

    def __init__(self, model_name: str = GENERATION_MODEL_NAME) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to("cpu")
        self.model.eval()

    @torch.inference_mode()
    def generate_answer(self, query: str, contexts: List[str]) -> str:
        context_text = "\n\n".join(contexts)

        prompt = (
            "You are a helpful assistant that answers questions using ONLY the information in the given Sanskrit passage.\n"
            "If the passage does not contain the answer, say clearly that the answer is not available in the text.\n"
            "Write the answer in simple, grammatical Sanskrit.\n\n"
            "पाठः (Sanskrit passage):\n"
            f"{context_text}\n\n"
            "प्रश्नः (Question):\n"
            f"{query}\n\n"
            "उत्तरम् (Answer in Sanskrit):\n"
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
        )

        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer.strip()

