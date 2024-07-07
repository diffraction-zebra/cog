# Prediction interface for Cog ⚙️
# https://cog.run/python
import os
from transformers import AutoModel, AutoTokenizer, HqqConfig
from cog import BasePredictor, Input, Path


class Predictor(BasePredictor):

    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = 'cuda'
        quant_config = HqqConfig(nbits=8, group_size=64, quant_zero=False, quant_scale=False,
                                 axis=0)
        model_name = 'openai-community/gpt2'
        self.model = AutoModel.from_pretrained(model_name, devicer_map=self.device, quantization_config=quant_config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def predict(
            self,
            prompt: str = "Let's describe how to write an website in 10 steps."
    ) -> str:
        tokens = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        pred = self.model(tokens)[0]
        answer = self.tokenizer.decode(pred, skip_special_tokens=True)
        return answer
