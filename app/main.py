
import argparse
import os
import json
import subprocess
import sys
import re
from pathlib import Path

import yaml
from openai import OpenAI


API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv(
    "OPENROUTER_BASE_URL",
    "https://openrouter.ai/api/v1",
)
MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "anthropic/claude-haiku-4.5",
)


def load_skills(skills_dir):
    skills = {}

    if not skills_dir.is_dir():
        return skills

    for skill_file in skills_dir.glob("*/SKILL.md"):
        content = skill_file.read_text(encoding="utf-8")
        parts = content.split("---", maxsplit=2)

        if (
            len(parts) != 3
            or parts[0].strip()
            or not parts[1].strip()
        ):
            continue

        try:
            metadata = yaml.safe_load(parts[1])
        except yaml.YAMLError:
            continue

        if not isinstance(metadata, dict):
            continue

        name = metadata.get("name")
        description = metadata.get("description")

        if not isinstance(name, str) or not isinstance(description, str):
            continue

        skills[name] = {
            "name": name,
            "description": description,
            "body": parts[2].strip(),
        }

    return skills


def substitute_arguments(body, arguments):
    # Replace indexed placeholders first.
    # The negative lookahead prevents $1 from matching $10.
    for index, argument in enumerate(arguments):
        body = body.replace(f"$ARGUMENTS[{index}]", argument)

    body = re.sub(
        r"\$(\d+)",
        lambda match: (
            arguments[int(match.group(1))]
            if int(match.group(1)) < len(arguments)
            else ""
        ),
        body,
    )

    # Replace $ARGUMENTS with the complete argument string.
    body = body.replace("$ARGUMENTS", " ".join(arguments))

    return body


def resolve_prompt(prompt, skills):
    if not prompt.startswith("/"):
        return prompt

    parts = prompt[1:].split(maxsplit=1)
    skill_name = parts[0]
    argument_string = parts[1] if len(parts) > 1 else ""

    if skill_name not in skills:
        raise ValueError(f"Unknown skill: {skill_name}")

    # Split arguments on whitespace, as required by the stage.
    arguments = argument_string.split()

    body = skills[skill_name]["body"]

    return substitute_arguments(body, arguments)


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

    skills = load_skills(Path(".claude/skills"))

    messages = []

    if skills:
        system_prompt = (
            "You have access to the following skills:\n\n"
            + "\n".join(
                f"- {skill['name']}: {skill['description']}"
                for skill in skills.values()
            )
        )
        messages.append({
            "role": "system",
            "content": system_prompt,
        })

    # Resolve slash command and substitute its arguments.
    user_prompt = resolve_prompt(args.p, skills)

    messages.append({
        "role": "user",
        "content": user_prompt,
    })

    tools = [
        {
            "type": "function",
            "function": {
                "name": "Read",
                "description": "Reads the content of a file",
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
                "description": "Writes the required content to a file",
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
            model=MODEL,
            messages=messages,
            tools=tools,
        )

        if not chat.choices:
            raise RuntimeError("No choices in response")

        message = chat.choices[0].message

        assistant_message = {
            "role": "assistant",
            "content": message.content,
        }

        if message.tool_calls:
            assistant_message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": call.type,
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in message.tool_calls
            ]

        messages.append(assistant_message)

        if not message.tool_calls:
            print(message.content or "")
            break

        for call in message.tool_calls:
            name = call.function.name
            kwargs = json.loads(call.function.arguments)

            try:
                if name == "Read":
                    result = Path(
                        kwargs["file_path"]
                    ).read_text(encoding="utf-8")

                elif name == "Write":
                    Path(kwargs["file_path"]).write_text(
                        kwargs["content"],
                        encoding="utf-8",
                    )
                    result = "OK"

                elif name == "Bash":
                    process = subprocess.run(
                        kwargs["command"],
                        shell=True,
                        capture_output=True,
                        text=True,
                    )
                    result = (
                        process.stdout
                        if process.stdout
                        else process.stderr
                    )

                else:
                    result = f"Unknown tool: {name}"

            except Exception as exc:
                result = f"Tool error: {exc}"

            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": result,
            })


if __name__ == "__main__":
    main()
