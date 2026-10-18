import json
from openai import OpenAI


def multiply(a: int, b: int) -> int:
    """Multiply two integers"""
    result = a * b
    return result


expected_argument_types = {"integer": int, "string": str, "boolean": bool}


def validate_tool_call(tool_call):
    """Check if a tool call is valid."""
    # Currently only support function tool calls
    if tool_call.type != "function":
        return False, None
    available_names = [
        available_tools[tool]["schema"]["function"]["name"] for tool in available_tools
    ]
    if tool_call.function.name not in available_names:
        return False, None
    schema = available_tools[tool_call.function.name]["schema"]

    try:
      args = json.loads(tool_call.function.arguments)
    except json.JSONDecodeError:
        print(f"Error: tool arguments not valid json")
        return False, None
    if not isinstance(args, dict):
        print(f"Error: tool call function arguments are not a dict")
        return False, None

    expected_params = schema["function"]["parameters"]["properties"]
    required_params = schema["function"]["parameters"]["required"]
    if len(required_params) > len(args):
        print(
            f"Error: tool {tool_call.function.name} requires {len(required_params)} parameters but received {len(args)}"
        )
        return False, None
    for req in required_params:
        if req not in args:
            print(f"Error: required argument {req} not in {args}")
            return False, None
        if args[req] == None:
            print(f"Error: required argument {req} should not be None")
            return False, None
    for arg in args:
        if arg not in expected_params:
            print(f"Error: argument {arg} not in expected parameters {expected_params}")
            return False, None
        if type(args[arg]) != expected_argument_types[expected_params[arg]["type"]]:
            print(
                f"Error: argument {arg} type is {type(args[arg])} but should be {expected_argument_types[expected_params[arg]['type']]}"
            )
            return False, None

    return True, args


multiply_schema = {
    "type": "function",
    "function": {
        "name": "multiply",
        "description": "Multiplies two integers",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "integer", "description": "The first factor"},
                "b": {"type": "integer", "description": "The second factor"},
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

message = response.choices[0].message
messages.append(message.model_dump(exclude_none=True))

tool_call = message.tool_calls[0]
valid, args = validate_tool_call(tool_call)
if not valid:
    print(f"Error: invalid tool call {tool_call}")
    exit()
func_to_call = available_tools[tool_call.function.name]["function"]

result = func_to_call(**args)

messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": f"{result}"})

response = client.chat.completions.create(
    model="qwen/qwen3.8-27b",
    messages=messages,
    tools=[available_tools[tool]["schema"] for tool in available_tools],
)

print(response.choices[0].message.content, flush=True)
