import argparse
import os
import json

from openai import OpenAI

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-p", required=True)
    args = p.parse_args()

    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    messages = [{"role": "user", "content": args.p}]
    while True:
        chat = client.chat.completions.create(
            model="anthropic/claude-haiku-4.5",
            messages=messages,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "Read",
                        "description": "Reads and displays the content of the file",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "file_path": {
                                    "type": "string",
                                    "description": "Gives the path of the file",
                                }
                            },
                            "required": ["file_path"],
                        },
                    },
                    },
                    {
                      "type": "function",
  "function": {
    "name": "Write",
    "description": "Write content to a file",
    "parameters": {
      "type": "object",
      "required": ["file_path", "content"],
      "properties": {
        "file_path": {
          "type": "string",
          "description": "The path of the file to write to"
        },
        "content": {
          "type": "string",
          "description": "The content to write to the file"
        }
      }
    }
  }
                },
            ],
        )
        if not chat.choices or len(chat.choices) == 0:
            raise RuntimeError("no choices in response")
        message = chat.choices[0].message
        assistant_message = {"role": "assistant", "content": message.content}
        if message.tool_calls:
            assistant_message["tool_calls"] = []
            for tool_call in message.tool_calls:
                assistant_message["tool_calls"].append(
                    {
                        "id": tool_call.id,
                        "type": tool_call.type,
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                )
        messages.append(assistant_message)
        if not message.tool_calls:
            print(message.content)
            break
        for tool_call in message.tool_calls:
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            result = ""
            if function_name == "Read":
                with open(arguments["file_path"], "r") as f:
                    result = f.read()
            elif function_name == "Write":
                with open(arguments["file_path"], "w") as f:
                    f.write(arguments["content"])
                result = f"Wrote {len(arguments['content'])} characters to {arguments['file_path']}"
            messages.append(
                {"role": "tool", "tool_call_id": tool_call.id, "content": result}
            )

if __name__ == "__main__":
    main()