import torch
from functools import lru_cache
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"

@lru_cache(maxsize=1)
def get_model_and_tokenizer(dtype=torch.bfloat16):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=dtype,          # transformers 5.x
        device_map="auto",
    )
    return model, tokenizer
