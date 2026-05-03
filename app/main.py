import argparse
import os
import sys

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

    chat = client.chat.completions.create(
        model="anthropic/claude-haiku-4.5",
        messages=[{"role": "user", "content": args.p},
                  {
  "role": "tool",
  "tool_call_id": "call_abc123",
  "content": "# My Project\n\nChemical expiry period: 6 months"
}],
        tools=[
            {
            "type":"function",
            "function":{
                "name":"Read",
                "dscription":"Reads and displays the content of the file",
                "parameters":{
                    "type":"object",
                    "properties":{
                        "file_path":{
                            "type":"string",
                            "description":"Gives the path of the file"
                        }
                    },
                    "required":["file_path"]
                }
            }
            }
        ]
    )
    if not chat.choices or len(chat.choices) == 0:
        raise RuntimeError("no choices in response")

    # You can use print statements as follows for debugging, they'll be visible when running tests.
    message = chat.choices[0].message
    if message.tool_calls:
        import json
        tool_call = message.tool_calls[0]
        function_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        if function_name == "Read":
            with open(arguments["file_path"], "r") as f:
                print(f.read(), end="")
    else:
        print(message.content)


    # TODO: Uncomment the following line to pass the first stage
    # print(chat.choices[0].message.content)


if __name__ == "__main__":
    main()
