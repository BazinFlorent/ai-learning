from threading import Thread
from transformers import TextIteratorStreamer
from llm.model import get_model_and_tokenizer

SYSTEM_PROMPT = "Tu es un assistant utile. Réponds uniquement en français."

def stream_answer(messages, max_new_tokens=512, temperature=0.7, top_p=0.9):
    model, tokenizer = get_model_and_tokenizer()

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    device = next(model.parameters()).device
    if hasattr(inputs, "items"):
        input_kwargs = {k: v.to(device) for k, v in inputs.items()}
    else:
        input_kwargs = {"input_ids": inputs.to(device)}

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_special_tokens=True,
        skip_prompt=True,
    )

    t = Thread(
        target=model.generate,
        kwargs=dict(
            **input_kwargs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        ),
    )
    t.start()

    for chunk in streamer:
        yield chunk

    t.join()
