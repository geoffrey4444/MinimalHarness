import lmstudio as lms

for model in lms.list_downloaded_models("llm"):
    print(model.model_key)
