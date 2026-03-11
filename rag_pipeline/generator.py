from __future__ import annotations

from transformers import pipeline


class Generator:
    def __init__(self, model_name: str = "google/flan-t5-base", device: int = -1):
        # self.generator = pipeline("text2text-generation", model=model_name, device=device)
        from transformers import pipeline, AutoModel, AutoConfig

        # Detect if model is T5/Flan-T5 and use the correct pipeline
        config = AutoConfig.from_pretrained(model_name)
        model_type = getattr(config, "model_type", "")
        if "t5" in model_type:
            self.generator = pipeline("text2text-generation", model=model_name, device=device)
            self._output_key = "generated_text"
        else:
            self.generator = pipeline("text-generation", model=model_name, device=device)
            self._output_key = "text"

    def generate(self, prompt: str, max_new_tokens: int = 180) -> str:
        result = self.generator(prompt, max_new_tokens=max_new_tokens, do_sample=False)
        return result[0][self._output_key].strip()
