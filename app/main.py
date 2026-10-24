
import argparse
import os
import json
import subprocess
from pathlib import Path
from openai import OpenAI


API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv(
    "OPENROUTER_BASE_URL",
    "https://openrouter.ai/api/v1",
)


def load_skill(skill_name):
    skill_path = Path(".claude") / "skills" / skill_name / "SKILL.md"

    with open(skill_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Extract the body after the YAML frontmatter.
    parts = content.split("---", 2)

    if len(parts) < 3:
        raise ValueError(
            f"Invalid SKILL.md frontmatter: {skill_path}"
        )

    return parts[2].strip()


def load_skill_descriptions():
    skills_dir = Path(".claude") / "skills"
    descriptions = []

    if skills_dir.is_dir():
        for skill_dir in sorted(skills_dir.iterdir()):
            if not skill_dir.is_dir():
                continue

            skill_path = skill_dir / "SKILL.md"

            if not skill_path.is_file():
                continue

            with open(skill_path, "r", encoding="utf-8") as f:
                content = f.read()

            parts = content.split("---", 2)

            if len(parts) < 3:
                continue

            metadata = {}

            for line in parts[1].splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    metadata[key.strip()] = value.strip().strip("\"'")

            name = metadata.get("name", skill_dir.name)
            description = metadata.get("description", "")

            descriptions.append(f"- {name}: {description}")

    return descriptions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", required=True)
    args = parser.parse_args()

    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
    )

    # List skill names and descriptions without loading their bodies.
    descriptions = load_skill_descriptions()
    system_prompt = "You have access to the following skills:\n"

    if descriptions:
        system_prompt += "\n" + "\n".join(descriptions)

    # Resolve slash commands and load only the selected skill body.
    user_prompt = args.p

    if user_prompt.startswith("/"):
        skill_name = user_prompt[1:].split(maxsplit=1)[0]
        user_prompt = load_skill(skill_name)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    tools = [
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
    ]

    while True:
        chat = client.chat.completions.create(
            model="anthropic/claude-haiku-4.5",
            messages=messages,
            tools=tools,
        )

        if not chat.choices:
            raise RuntimeError("No choices in response")

        message = chat.choices[0].message

        # Preserve assistant content and tool calls in conversation history.
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

        # Finish when the model does not request a tool.
        if not message.tool_calls:
            print(message.content or "")
            break

        # Execute requested tools and return their results.
        for tc in message.tool_calls:
            name = tc.function.name
            tool_args = json.loads(tc.function.arguments)

            result = ""

            if name == "Read":
                try:
                    with open(
                        tool_args["file_path"],
                        "r",
                        encoding="utf-8",
                    ) as f:
                        result = f.read()
                except Exception as e:
                    result = f"Error reading file: {e}"

            elif name == "Write":
                try:
                    with open(
                        tool_args["file_path"],
                        "w",
                        encoding="utf-8",
                    ) as f:
                        f.write(tool_args["content"])
                    result = "OK"
                except Exception as e:
                    result = f"Error writing file: {e}"

            elif name == "Bash":
                try:
                    proc = subprocess.run(
                        tool_args["command"],
                        shell=True,
                        capture_output=True,
                        text=True,
                    )
                    result = (
                        proc.stdout
                        if proc.stdout
                        else proc.stderr
                    )
                except Exception as e:
                    result = str(e)

            else:
                result = f"Unknown tool: {name}"

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })


if __name__ == "__main__":
    main()
