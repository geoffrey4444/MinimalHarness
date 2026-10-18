import lmstudio as lms

model = lms.llm(
    "qwen/qwen3.8-27b", config={"contextLength": 128000, "gpu": {"ratio": 1.0}}
)
chat = lms.Chat("You are a helpful assistant.")


def multiply(a: int, b: int) -> int:
    """Multiply two integers"""
    result = a * b
    print(f"\n[TOOL] multiply({a}, {b}) = {result}")
    return result


def print_fragment(fragment, round_index):
    print(fragment.content, end="", flush=True)


while True:
    prompt = input("\n> ")
    if prompt.lower() in ("quit", "exit", ""):
        break

    chat.add_user_message(prompt)

    model.act(
        chat, [multiply], on_message=chat.append, on_prediction_fragment=print_fragment
    )
