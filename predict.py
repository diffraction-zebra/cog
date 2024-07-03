# Prediction interface for Cog ⚙️
# https://cog.run/python

import requests
import subprocess

from cog import BasePredictor, Input, Path


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # self.model = torch.load("./weights.pth")
        subprocess.Popen(["/llama.cpp/llama-server", "-m", "/models/gpt2.Q8_0.gguf", "-c", "2048"])

    def predict(
            self,
            prompt: str = "Let's describe how to write an website in 10 steps."
    ) -> str:
        url = 'http://localhost:8080/completion'
        headers = {"Content-Type": "application/json"}
        data = {"prompt": prompt, "n_predict": 128}
        req = requests.post(url, headers=headers, data=data)

        return req.text
