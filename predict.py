# Prediction interface for Cog ⚙️
# https://cog.run/python
import os
import pathlib

from transformers import AutoModel, AutoTokenizer
from cog import BasePredictor


class Predictor(BasePredictor):

    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = 'cuda'
        model_path = pathlib.Path('model')
        self.model = AutoModel.from_pretrained(model_path, device_map=self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def predict(
            self,
            prompt: str = "Let's describe how to write an website in 10 steps."
    ) -> str:
        messages = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                       return_tensors="pt").to(self.device)

        gen_tokens = self.model.generate(
            input_ids,
            max_new_tokens=4000,
            do_sample=True,
            temperature=0.3,
        )

        gen_text = self.tokenizer.decode(gen_tokens[0])
        return gen_text
