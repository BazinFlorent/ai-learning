from llm.chat import stream_answer, SYSTEM_PROMPT

def main():
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        user_input = input("\nVous : ").strip()
        if user_input.lower() in {"exit", "quit", "q"}:
            break

        messages.append({"role": "user", "content": user_input})

        print("Assistant : ", end="", flush=True)
        response = ""
        for chunk in stream_answer(messages):
            print(chunk, end="", flush=True)
            response += chunk
        print()

        messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
