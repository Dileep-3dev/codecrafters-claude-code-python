
import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
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


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    path: Path
    body: str


@dataclass
class Skills:
    skills: dict[str, Skill]

    @staticmethod
    def load(path: Path) -> "Skills":
        skills = {}

        if not path.is_dir():
            return Skills(skills={})

        for skill_path in sorted(path.glob("*/SKILL.md")):
            try:
                content = skill_path.read_text(encoding="utf-8")
            except OSError as exc:
                print(
                    f"WARN: Cannot read {skill_path}: {exc}",
                    file=sys.stderr,
                )
                continue

            sections = content.split("---", maxsplit=2)

            if (
                len(sections) != 3
                or sections[0].strip()
                or not sections[1].strip()
            ):
                print(
                    f"WARN: Invalid frontmatter in {skill_path}",
                    file=sys.stderr,
                )
                continue

            try:
                metadata = yaml.safe_load(sections[1])
            except yaml.YAMLError as exc:
                print(
                    f"WARN: Invalid YAML in {skill_path}: {exc}",
                    file=sys.stderr,
                )
                continue

            if not isinstance(metadata, dict):
                print(
                    f"WARN: Invalid metadata in {skill_path}",
                    file=sys.stderr,
                )
                continue

            name = metadata.get("name")
            description = metadata.get("description")

            if not isinstance(name, str) or not name.strip():
                print(
                    f"WARN: Missing name in {skill_path}",
                    file=sys.stderr,
                )
                continue

            if not isinstance(description, str) or not description.strip():
                print(
                    f"WARN: Missing description in {skill_path}",
                    file=sys.stderr,
                )
                continue

            skills[name] = Skill(
                name=name,
                description=description,
                path=skill_path,
                body=sections[2].strip(),
            )

        return Skills(skills=skills)

    def is_empty(self) -> bool:
        return not self.skills

    def level1_system_prompt(self) -> str:
        lines = [
            "You have access to the following skills:",
            "",
        ]

        lines.extend(
            f"- {skill.name}: {skill.description}"
            for skill in self.skills.values()
        )

        return "\n".join(lines)


def expand_skill_arguments(body: str, arguments: list[str]) -> str:
    # Replace indexed placeholders first, before $ARGUMENTS.
    for index, argument in enumerate(arguments):
        body = body.replace(f"$ARGUMENTS[{index}]", argument)
        body = re.sub(
            rf"\${index}(?!\d)",
            lambda _: argument,
            body,
        )

    # Replace $ARGUMENTS with all supplied arguments.
    body = body.replace("$ARGUMENTS", " ".join(arguments))

    return body


def resolve_prompt(prompt: str, skills: Skills) -> str:
    if not prompt.startswith("/"):
        return prompt

    command_parts = prompt[1:].split(maxsplit=1)
    skill_name = command_parts[0]
    argument_text = command_parts[1] if len(command_parts) > 1 else ""

    skill = skills.skills.get(skill_name)

    if skill is None:
        available = ", ".join(skills.skills.keys()) or "(none)"
        raise ValueError(
            f"Unknown skill: {skill_name}. Available skills: {available}"
        )

    # Split arguments on whitespace, as in the example implementation.
    arguments = argument_text.split()

    return expand_skill_arguments(skill.body, arguments)


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

    # Load skill metadata and bodies into memory.
    # Only the selected skill's body will be sent to the model.
    skills = Skills.load(Path(".claude/skills"))

    messages = []

    if not skills.is_empty():
        system_prompt = skills.level1_system_prompt()
        messages.append({
            "role": "system",
            "content": system_prompt,
        })

        print(
            f"=== SYSTEM PROMPT ===\n"
            f"{system_prompt}\n"
            f"=====================",
            file=sys.stderr,
        )

    prompt = resolve_prompt(args.p, skills)
    messages.append({
        "role": "user",
        "content": prompt,
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
            return

        for call in message.tool_calls:
            tool_name = call.function.name
            kwargs = json.loads(call.function.arguments)

            formatted_args = ", ".join(
                f'{key}="{value}"'
                for key, value in kwargs.items()
            )

            print(
                f"> {tool_name}({formatted_args})",
                file=sys.stderr,
            )

            result = ""

            try:
                if tool_name == "Read":
                    result = Path(
                        kwargs["file_path"]
                    ).read_text(encoding="utf-8")

                elif tool_name == "Write":
                    Path(kwargs["file_path"]).write_text(
                        kwargs["content"],
                        encoding="utf-8",
                    )
                    result = "OK"

                elif tool_name == "Bash":
                    process = subprocess.run(
                        kwargs["command"],
                        capture_output=True,
                        text=True,
                        shell=True,
                    )
                    result = (
                        process.stdout
                        if process.stdout
                        else process.stderr
                    )

                else:
                    result = f"Unknown tool: {tool_name}"

            except Exception as exc:
                result = f"Tool error: {exc}"

            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": result,
            })


if __name__ == "__main__":
    main()
