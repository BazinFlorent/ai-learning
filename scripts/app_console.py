from threading import Thread
from transformers import TextIteratorStreamer
from llm.model import get_model_and_tokenizer

def main():
    model, tokenizer = get_model_and_tokenizer()

    messages = [
        {"role": "system", "content": "Tu es un assistant utile. Réponds uniquement en français."}
    ]

    while True:
        user_input = input("\nVous : ").strip()

        if user_input.lower() in {"exit", "quit", "q"}:
            break

        messages.append({
            "role": "user",
            "content": user_input
        })

        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )

        # Mettre les entrées sur le même device que le modèle (au moins le premier shard)
        device = next(model.parameters()).device

        if hasattr(inputs, "items"):
            inputs = {k: v.to(device) for k, v in inputs.items()}
            input_kwargs = inputs
        else:
            # cas rare: tensor direct (input_ids)
            input_kwargs = {"input_ids": inputs.to(device)}

        streamer = TextIteratorStreamer(
            tokenizer,
            skip_special_tokens=True,
            skip_prompt=True,
        )

        gen_kwargs = dict(
            **input_kwargs,
            streamer=streamer,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        # generate() bloque: on le lance dans un thread
        t = Thread(target=model.generate, kwargs=gen_kwargs)
        t.start()

        # Affichage progressif en console
        response = ""
        for chunk in streamer:
            print(chunk, end="", flush=True)
            response += chunk

        t.join()
        print()  # newline final

        messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
