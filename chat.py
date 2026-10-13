import lmstudio as lms

model = lms.llm("qwen/qwen3.8-27b")
chat = lms.Chat("You are a helpful assistant.")

while True:
    prompt = input("> ")
    if prompt.lower() in ("quit", "exit"):
        break

    chat.add_user_message(prompt)

    response = model.respond(chat)
    chat.add_assistant_response(response)

    print(response)
