import lmstudio as lms

model = lms.llm(
    "qwen/qwen3.8-27b", config={"contextLength": 128000, "gpu": {"ratio": 1.0}}
)
chat = lms.Chat("You are a helpful assistant.")

while True:
    prompt = input("> ")
    if prompt.lower() in ("quit", "exit"):
        break

    chat.add_user_message(prompt)

    response = ""
    for fragment in model.respond_stream(chat):
        print(fragment.content, end="", flush=True)
        response += fragment.content

    chat.add_assistant_response(response)

    print(response)
