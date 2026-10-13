import lmstudio as lms

for model in lms.list_downloaded_models("llm"):
    print(model.model_key)

model = lms.llm("qwen/qwen3.8-27b")

response = model.respond("Explain numerical relativity in one paragraph.")

print(response)
