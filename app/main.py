
import argparse
import json
import os
import re
import shlex
import subprocess
from pathlib import Path

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
    """Load skill metadata and bodies without external YAML dependencies."""
    skills = {}

    if not skills_dir.is_dir():
        return skills

    for skill_file in sorted(skills_dir.glob("*/SKILL.md")):
        try:
            content = skill_file.read_text(encoding="utf-8")
        except OSError:
            continue

        parts = content.split("---", maxsplit=2)

        if len(parts) != 3 or parts[0].strip():
            continue

        metadata = {}

        for line in parts[1].splitlines():
            if ":" not in line:
                continue

            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip().strip("\"'")

        name = metadata.get("name")
        description = metadata.get("description")

        if not name or not description:
            continue

        skills[name] = {
            "name": name,
            "description": description,
            "body": parts[2].strip(),
        }

    return skills


def substitute_arguments(body, arguments):
    """Replace skill argument placeholders."""
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

    body = body.replace("$ARGUMENTS", " ".join(arguments))
    return body


def resolve_prompt(prompt, skills):
    """Expand consecutive skill commands typed by the user."""
    tokens = prompt.split()
    expanded_bodies = []
    index = 0

    while index < len(tokens):
        token = tokens[index]

        if not token.startswith("/") or len(token) == 1:
            break

        skill_name = token[1:]

        if skill_name not in skills:
            break

        expanded_bodies.append(skills[skill_name]["body"])
        index += 1

    if not expanded_bodies:
        return [{"role": "user", "content": prompt}]

    arguments = tokens[index:]

    return [
        {
            "role": "user",
            "content": substitute_arguments(body, arguments),
        }
        for body in expanded_bodies
    ]


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

    skills = load_skills(Path(".claude") / "skills")
    messages = []

    if skills:
        system_prompt = (
            "You have access to the following skills:\n\n"
            + "\n".join(
                f"- {skill['name']}: {skill['description']}"
                for skill in skills.values()
            )
            + "\n\n"
            "If a skill matches the user's request, call the Skill tool "
            "with its name and follow the instructions it returns."
        )

        messages.append({
            "role": "system",
            "content": system_prompt,
        })

    messages.extend(resolve_prompt(args.p, skills))

    tools = [
        {
            "type": "function",
            "function": {
                "name": "Read",
                "description": "Reads the content of a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
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
                        "command": {"type": "string"},
                    },
                    "required": ["command"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "Skill",
                "description": "Load a skill's instructions into the conversation",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "The name of the skill to use",
                        },
                        "args": {
                            "type": "string",
                            "description": "Optional arguments for the skill",
                        },
                    },
                    "required": ["name"],
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

        # Preserve the assistant response and its tool calls.
        assistant_message = {
            "role": "assistant",
            "content": message.content,
        }

        if message.tool_calls:
            assistant_message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
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

            try:
                kwargs = json.loads(call.function.arguments)

                if name == "Skill":
                    skill_name = kwargs.get("name", "")
                    skill_args = kwargs.get("args", "")

                    if skill_name not in skills:
                        result = f"Error: Skill '{skill_name}' not found."
                    else:
                        arguments = shlex.split(skill_args)
                        result = substitute_arguments(
                            skills[skill_name]["body"],
                            arguments,
                        )

                elif name == "Read":
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
