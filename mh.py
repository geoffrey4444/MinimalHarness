import json
from openai import OpenAI


def multiply(a: int, b: int) -> int:
    """Multiply two integers"""
    result = a * b
    print(f"\n[TOOL] multiply({a}, {b}) = {result}")
    return result


expected_argument_types = {"int": int}


def validate_tool_call(tool_call):
    """Check if a tool call is valid."""
    # Currently only support function tool calls
    if tool_call.type != "function":
        return False
    available_names = [
        available_tools[tool]["schema"]["function"]["name"] for tool in available_tools
    ]
    if tool_call.function.name not in available_names:
        return False
    schema = available_tools[tool_call.function.name]["schema"]
    args = json.loads(tool_call.function.arguments)
    expected_params = schema["function"]["parameters"]["properties"]
    required_params = schema["function"]["parameters"]["required"]
    if len(args) > len(expected_params):
        print(
            f"Error: tool {tool_call.function.name} accepts {len(expected_params)} parameters but received {len(args)}"
        )
        return False
    if len(required_params) > len(args):
        print(
            f"Error: tool {tool_call.function.name} requires {len(required_params)} parameters but received {len(args)}"
        )
        return False
    for req in required_params:
        if req not in args:
            print(f"Error: required argument {req} not in {args}")
            return False
    for arg in args:
        if arg not in expected_params:
            print(f"Error: argument {arg} not in expected parameters {expected_params}")
            return False
        if type(args[arg]) != expected_argument_types[expected_params[arg]["type"]]:
            print(
                f"Error: argument {arg} type is {type(args[arg])} but should be {expected_argument_types[expected_params[arg]['type']]}"
            )
            return False

    return True


multiply_schema = {
    "type": "function",
    "function": {
        "name": "multiply",
        "description": "Multiplies two integers",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "int", "description": "The first factor"},
                "b": {"type": "int", "description": "The second factor"},
            },
            "required": ["a", "b"],
        },
    },
}

available_tools = {"multiply": {"function": multiply, "schema": multiply_schema}}


client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Use multiplty to calculate 432*567"},
]

response = client.chat.completions.create(
    model="qwen/qwen3.8-27b",
    messages=messages,
    tools=[available_tools[tool]["schema"] for tool in available_tools],
)
messages.append({"role": "assistant", "content": str(response)})

message = response.choices[0].message
tool_call = message.tool_calls[0]
if not validate_tool_call(tool_call):
    print(f"Error: invalid tool call {tool_call}")
    exit()

func_to_call = available_tools[tool_call.function.name]["function"]
print(type(tool_call.function.arguments), flush=True)
print(tool_call.function.arguments, flush=True)
args = json.loads(tool_call.function.arguments)
print(args, flush=True)
result = func_to_call(**args)

messages.append(
    {
        "role": "assistant",
        "content": f"Tool {tool_call.function.name} called with arguments {args}",
    }
)
messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": f"{result}"})

response = client.chat.completions.create(
    model="qwen/qwen3.8-27b",
    messages=messages,
    tools=[available_tools[tool]["schema"] for tool in available_tools],
)

print(response.choices[0].message.content, flush=True)
