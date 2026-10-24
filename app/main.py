import argparse
import os
import json
import subprocess
from openai import OpenAI

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")


def load_skill(skill_name):
    skill_path = os.path.join(
        ".claude", "skills", skill_name, "SKILL.md"
    )

    with open(skill_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Remove the YAML frontmatter and return only the body
    parts = content.split("---", 2)

    if len(parts) < 3:
        raise ValueError("Invalid SKILL.md: missing frontmatter")

    return parts[2].strip()


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
                        "description": "Reads the content of the file",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "file_path": {"type": "string"}
                            },
                            "required": ["file_path"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "Write",
                        "description": "Writes the required content to the file",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "file_path": {"type": "string"},
                                "content": {"type": "string"},
                            },
                            "required": ["file_path", "content"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "Bash",
                        "description": "Execute a shell command",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "command": {"type": "string"}
                            },
                            "required": ["command"],
                        },
                    },
                },
            ],
        )

        if not chat.choices:
            raise RuntimeError("no choices in response")

        message = chat.choices[0].message

        # Add assistant message
        assistant_msg = {
            "role": "assistant",
            "content": message.content,
        }

        if message.tool_calls:
            assistant_msg["tool_calls"] = []
            for tc in message.tool_calls:
                assistant_msg["tool_calls"].append({
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                })

        messages.append(assistant_msg)

        # If no tools → done
        if not message.tool_calls:
            print(message.content)
            break

        # Execute tools
        for tc in message.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments)

            result = ""

            if name == "Read":
                with open(args["file_path"], "r") as f:
                    result = f.read()

            elif name == "Write":
                with open(args["file_path"], "w") as f:
                    f.write(args["content"])
                result = "OK"

            elif name == "Bash":
                try:
                    proc = subprocess.run(
                        args["command"],
                        shell=True,
                        capture_output=True,
                        text=True,
                    )
                    result = proc.stdout if proc.stdout else proc.stderr
                except Exception as e:
                    result = str(e)

            # Send tool result back
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })


if __name__ == "__main__":
    main()