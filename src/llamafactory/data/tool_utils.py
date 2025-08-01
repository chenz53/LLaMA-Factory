# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, NamedTuple, Union

from typing_extensions import override


class FunctionCall(NamedTuple):
    name: str
    arguments: str


DEFAULT_TOOL_PROMPT = (
    "You have access to the following tools:\n{tool_text}"
    "Use the following format if using a tool:\n"
    "```\n"
    "Action: tool name (one of [{tool_names}])\n"
    "Action Input: the input to the tool, in a JSON format representing the kwargs "
    """(e.g. ```{{"input": "hello world", "num_beams": 5}}```)\n"""
    "```\n"
)

GLM4_TOOL_PROMPT = (
    "你是一个名为 ChatGLM 的人工智能助手。你是基于智谱 AI 公司训练的语言模型 GLM-4 模型开发的，"
    "你的任务是针对用户的问题和要求提供适当的答复和支持。\n\n# 可用工具{tool_text}"
)

GLM4_MOE_TOOL_PROMPT = (
    "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n<tools>{tool_text}"
    "\n</tools>\n\nFor each function call, output the function name and arguments within the following XML format:"
    "\n<tool_call>{{function-name}}"
    "\n<arg_key>{{arg-key-1}}</arg_key>"
    "\n<arg_value>{{arg-value-1}}</arg_value>"
    "\n<arg_key>{{arg-key-2}}</arg_key>"
    "\n<arg_value>{{arg-value-2}}</arg_value>"
    "\n...\n</tool_call>\n"
)

LLAMA3_TOOL_PROMPT = (
    "Cutting Knowledge Date: December 2023\nToday Date: {date}\n\n"
    "You have access to the following functions. To call a function, please respond with JSON for a function call. "
    """Respond in the format {{"name": function name, "parameters": dictionary of argument name and its value}}. """
    "Do not use variables.\n\n{tool_text}"
)

QWEN_TOOL_PROMPT = (
    "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n<tools>{tool_text}"
    "\n</tools>\n\nFor each function call, return a json object with function name and arguments within "
    """<tool_call></tool_call> XML tags:\n<tool_call>\n{{"name": <function-name>, """
    """"arguments": <args-json-object>}}\n</tool_call>"""
)


@dataclass
class ToolUtils(ABC):
    """Base class for tool utilities."""

    @staticmethod
    @abstractmethod
    def tool_formatter(tools: list[dict[str, Any]]) -> str:
        r"""Generate the system message describing all the available tools."""
        ...

    @staticmethod
    @abstractmethod
    def function_formatter(functions: list["FunctionCall"]) -> str:
        r"""Generate the assistant message including all the tool calls."""
        ...

    @staticmethod
    @abstractmethod
    def tool_extractor(content: str) -> Union[str, list["FunctionCall"]]:
        r"""Extract all the function calls from the assistant message.

        It should be an inverse function of `function_formatter`.
        """
        ...


class DefaultToolUtils(ToolUtils):
    r"""Default tool using template."""

    @override
    @staticmethod
    def tool_formatter(tools: list[dict[str, Any]]) -> str:
        tool_text = ""
        tool_names = []
        for tool in tools:
            tool = tool.get("function", "") if tool.get("type") == "function" else tool
            param_text = ""
            for name, param in tool["parameters"]["properties"].items():
                required, enum, items = "", "", ""
                if name in tool["parameters"].get("required", []):
                    required = ", required"

                if param.get("enum", None):
                    enum = ", should be one of [{}]".format(", ".join(param["enum"]))

                if param.get("items", None):
                    items = ", where each item should be {}".format(param["items"].get("type", ""))

                param_text += "  - {name} ({type}{required}): {desc}{enum}{items}\n".format(
                    name=name,
                    type=param.get("type", ""),
                    required=required,
                    desc=param.get("description", ""),
                    enum=enum,
                    items=items,
                )

            tool_text += "> Tool Name: {name}\nTool Description: {desc}\nTool Args:\n{args}\n".format(
                name=tool["name"], desc=tool.get("description", ""), args=param_text
            )
            tool_names.append(tool["name"])

        return DEFAULT_TOOL_PROMPT.format(tool_text=tool_text, tool_names=", ".join(tool_names))

    @override
    @staticmethod
    def function_formatter(functions: list["FunctionCall"]) -> str:
        return "\n".join([f"Action: {name}\nAction Input: {arguments}" for name, arguments in functions])

    @override
    @staticmethod
    def tool_extractor(content: str) -> Union[str, list["FunctionCall"]]:
        regex = re.compile(r"Action:\s*([a-zA-Z0-9_]+)\s*Action Input:\s*(.+?)(?=\s*Action:|\s*$)", re.DOTALL)
        action_match: list[tuple[str, str]] = re.findall(regex, content)
        if not action_match:
            return content

        results = []
        for match in action_match:
            tool_name = match[0].strip()
            tool_input = match[1].strip().strip('"').strip("```")
            try:
                arguments = json.loads(tool_input)
                results.append(FunctionCall(tool_name, json.dumps(arguments, ensure_ascii=False)))
            except json.JSONDecodeError:
                return content

        return results


class GLM4ToolUtils(ToolUtils):
    r"""GLM-4 tool using template."""

    @override
    @staticmethod
    def tool_formatter(tools: list[dict[str, Any]]) -> str:
        tool_text = ""
        for tool in tools:
            tool = tool.get("function", "") if tool.get("type") == "function" else tool
            tool_text += "\n\n## {name}\n\n{body}\n在调用上述函数时，请使用 Json 格式表示调用的参数。".format(
                name=tool["name"], body=json.dumps(tool, indent=4, ensure_ascii=False)
            )

        return GLM4_TOOL_PROMPT.format(tool_text=tool_text)

    @override
    @staticmethod
    def function_formatter(functions: list["FunctionCall"]) -> str:
        if len(functions) > 1:
            raise ValueError("GLM-4 does not support parallel functions.")

        return f"{functions[0].name}\n{functions[0].arguments}"

    @override
    @staticmethod
    def tool_extractor(content: str) -> Union[str, list["FunctionCall"]]:
        if "\n" not in content:
            return content

        tool_name, tool_input = content.split("\n", maxsplit=1)
        try:
            arguments = json.loads(tool_input.strip())
        except json.JSONDecodeError:
            return content

        return [FunctionCall(tool_name, json.dumps(arguments, ensure_ascii=False))]


class Llama3ToolUtils(ToolUtils):
    r"""Llama 3.x tool using template with `tools_in_user_message=False`.

    Reference: https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_1/#json-based-tool-calling
    """

    @override
    @staticmethod
    def tool_formatter(tools: list[dict[str, Any]]) -> str:
        date = datetime.now().strftime("%d %b %Y")
        tool_text = ""
        for tool in tools:
            wrapped_tool = tool if tool.get("type") == "function" else {"type": "function", "function": tool}
            tool_text += json.dumps(wrapped_tool, indent=4, ensure_ascii=False) + "\n\n"

        return LLAMA3_TOOL_PROMPT.format(date=date, tool_text=tool_text)

    @override
    @staticmethod
    def function_formatter(functions: list["FunctionCall"]) -> str:
        function_objects = [{"name": name, "parameters": json.loads(arguments)} for name, arguments in functions]
        return json.dumps(function_objects[0] if len(function_objects) == 1 else function_objects, ensure_ascii=False)

    @override
    @staticmethod
    def tool_extractor(content: str) -> Union[str, list["FunctionCall"]]:
        try:
            tools = json.loads(content.strip())
        except json.JSONDecodeError:
            return content

        tools = [tools] if not isinstance(tools, list) else tools
        try:
            return [FunctionCall(tool["name"], json.dumps(tool["parameters"], ensure_ascii=False)) for tool in tools]
        except KeyError:
            return content


class MistralToolUtils(ToolUtils):
    r"""Mistral v0.3 tool using template."""

    @override
    @staticmethod
    def tool_formatter(tools: list[dict[str, Any]]) -> str:
        wrapped_tools = []
        for tool in tools:
            wrapped_tools.append(tool if tool.get("type") == "function" else {"type": "function", "function": tool})

        return "[AVAILABLE_TOOLS] " + json.dumps(wrapped_tools, ensure_ascii=False) + "[/AVAILABLE_TOOLS]"

    @override
    @staticmethod
    def function_formatter(functions: list["FunctionCall"]) -> str:
        return json.dumps(
            [{"name": name, "arguments": json.loads(arguments)} for name, arguments in functions], ensure_ascii=False
        )

    @override
    @staticmethod
    def tool_extractor(content: str) -> Union[str, list["FunctionCall"]]:
        try:
            tools = json.loads(content.strip())
        except json.JSONDecodeError:
            return content

        tools = [tools] if not isinstance(tools, list) else tools
        try:
            return [FunctionCall(tool["name"], json.dumps(tool["arguments"], ensure_ascii=False)) for tool in tools]
        except KeyError:
            return content


class QwenToolUtils(ToolUtils):
    r"""Qwen 2.5 tool using template."""

    @override
    @staticmethod
    def tool_formatter(tools: list[dict[str, Any]]) -> str:
        tool_text = ""
        for tool in tools:
            wrapped_tool = tool if tool.get("type") == "function" else {"type": "function", "function": tool}
            tool_text += "\n" + json.dumps(wrapped_tool, ensure_ascii=False)

        return QWEN_TOOL_PROMPT.format(tool_text=tool_text)

    @override
    @staticmethod
    def function_formatter(functions: list["FunctionCall"]) -> str:
        function_texts = [
            json.dumps({"name": name, "arguments": json.loads(arguments)}, ensure_ascii=False)
            for name, arguments in functions
        ]
        return "\n".join([f"<tool_call>\n{text}\n</tool_call>" for text in function_texts])

    @override
    @staticmethod
    def tool_extractor(content: str) -> Union[str, list["FunctionCall"]]:
        regex = re.compile(r"<tool_call>(.+?)</tool_call>(?=\s*<tool_call>|\s*$)", re.DOTALL)
        tool_match: list[str] = re.findall(regex, content)
        if not tool_match:
            return content

        results = []
        for tool in tool_match:
            try:
                tool = json.loads(tool.strip())
            except json.JSONDecodeError:
                return content

            if "name" not in tool or "arguments" not in tool:
                return content

            results.append(FunctionCall(tool["name"], json.dumps(tool["arguments"], ensure_ascii=False)))

        return results


class GLM4MOEToolUtils(QwenToolUtils):
    r"""GLM-4-MOE tool using template."""

    @override
    @staticmethod
    def tool_formatter(tools: list[dict[str, Any]]) -> str:
        tool_text = ""
        for tool in tools:
            wrapped_tool = tool if tool.get("type") == "function" else {"type": "function", "function": tool}
            tool_text += "\n" + json.dumps(wrapped_tool, ensure_ascii=False)

        return GLM4_MOE_TOOL_PROMPT.format(tool_text=tool_text)

    @override
    @staticmethod
    def function_formatter(functions: list["FunctionCall"]) -> str:
        function_json = [
            {"func_name": name, "func_key_values": json.loads(arguments)} for name, arguments in functions
        ]
        function_texts = []
        for func in function_json:
            prompt = "\n<tool_call>" + func["func_name"]
            for key, value in func["func_key_values"].items():
                prompt += "\n<arg_key>" + key + "</arg_key>"
                if not isinstance(value, str):
                    value = json.dumps(value, ensure_ascii=False)
                prompt += "\n<arg_value>" + value + "</arg_value>"
            function_texts.append(prompt)

        return "\n".join(function_texts)


TOOLS = {
    "default": DefaultToolUtils(),
    "glm4": GLM4ToolUtils(),
    "llama3": Llama3ToolUtils(),
    "mistral": MistralToolUtils(),
    "qwen": QwenToolUtils(),
    "glm4_moe": GLM4MOEToolUtils(),
}


def get_tool_utils(name: str) -> "ToolUtils":
    tool_utils = TOOLS.get(name, None)
    if tool_utils is None:
        raise ValueError(f"Tool utils `{name}` not found.")

    return tool_utils
