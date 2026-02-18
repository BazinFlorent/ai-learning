import gradio as gr
from llm.chat import stream_answer, SYSTEM_PROMPT

def _content_to_str(content):
    # Gradio 6 peut fournir:
    # - str
    # - list (segments)
    # - dict (segment)
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                # formats fréquents: {"type":"text","text":"..."} ou {"text":"..."}
                if "text" in item and isinstance(item["text"], str):
                    parts.append(item["text"])
                elif "content" in item:
                    parts.append(_content_to_str(item["content"]))
                else:
                    parts.append(str(item))
            else:
                parts.append(str(item))
        return "".join(parts)
    if isinstance(content, dict):
        if "text" in content and isinstance(content["text"], str):
            return content["text"]
        if "content" in content:
            return _content_to_str(content["content"])
        return str(content)
    return str(content)

def respond(user_message, history):
    history = history or []

    # On reconstruit un historique "propre" (role=user/assistant, content=str)
    cleaned_history = []
    for msg in history:
        role = msg.get("role")
        if role not in ("user", "assistant"):
            continue
        cleaned_history.append({
            "role": role,
            "content": _content_to_str(msg.get("content"))
        })

    user_message = _content_to_str(user_message)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + cleaned_history
    messages.append({"role": "user", "content": user_message})

    new_history = cleaned_history + [{"role": "user", "content": user_message}]

    partial = ""
    for chunk in stream_answer(messages):
        partial += chunk
        # On renvoie un historique compatible Chatbot (dict role/content)
        yield new_history + [{"role": "assistant", "content": partial}], ""

with gr.Blocks() as demo:
    gr.Markdown("# Chat local (Mistral 7B)")

    chat = gr.Chatbot(height=520)
    msg = gr.Textbox(label="Votre message", placeholder="Tapez ici…")
    clear = gr.Button("Clear")

    msg.submit(respond, [msg, chat], [chat, msg])
    clear.click(lambda: ([], ""), None, [chat, msg])

demo.queue()
demo.launch(inbrowser=True, prevent_thread_lock=False, debug=True, show_error=True)
