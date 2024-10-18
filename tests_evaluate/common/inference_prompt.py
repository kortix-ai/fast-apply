# v13
template = """Merge all changes from the <update> snippet into the <code> below, ensuring every modification is fully integrated.
- Preserve the code's structure, order, comments, and indentation exactly.
- Output only the updated code, enclosed within <updated-code> and </updated-code> tags.
- Do not include any additional text, explanations, placeholders, ellipses, or code fences.
\n
<code>{original_code}</code>
\n
<update>{update_snippet}</update>
\n
Here's the complete updated code:
"""

simple_template = """
Your task is to merge all changes from the update snippet into the original code, ensuring that every modification is fully integrated.
Maintain the code's structure, order, comments, and indentation precisely.
Do not use any placeholders, ellipses, or omit any sections.
Only output the updated code; do not include any additional text or explanations.

### Original Code:
{original_code}

### Update Snippet:
{update_snippet}

### Updated Code:
"""

##############################################################################

# SYSTEM_PROMPT = """You are an coding assistant that helps merge code updates. Merge all changes from the <update> snippet into the <code> below, ensuring every modification is fully integrated.
# - Preserve the code's structure, order, comments, and indentation exactly.
# - Output only the updated code, enclosed within <updated-code> and </updated-code> tags.
# - Do not include any additional text, explanations, placeholders, ellipses, or code fences."""

# USER_PROMPT = """<code>{original_code}</code>

# <update>{update_snippet}</update>

# Provide the complete updated code.
# """

SYSTEM_PROMPT = """You are an coding assistant that helps merge code updates, ensuring every modification is fully integrated."""

USER_PROMPT = """Merge all changes from the <update> snippet into the <code> below.
- Preserve the code's structure, order, comments, and indentation exactly.
- Output only the updated code, enclosed within <updated-code> and </updated-code> tags.
- Do not include any additional text, explanations, placeholders, ellipses, or code fences.

<code>{original_code}</code>

<update>{update_snippet}</update>

Provide the complete updated code.
"""


# for vLLM config
chat_template = """{%- if tools %}
    {{- \'<|im_start|>system\\n\' }}
    {%- if messages[0][\'role\'] == \'system\' %}
        {{- messages[0][\'content\'] }}
    {%- else %}
        {{- "You are an assistant that helps merge code updates. Merge all changes from the <update> snippet into the <code> below, ensuring every modification is fully integrated.
- Preserve the code's structure, order, comments, and indentation exactly.
- Output only the updated code, enclosed within <updated-code> and </updated-code> tags.
- Do not include any additional text, explanations, placeholders, ellipses, or code fences." }}
    {%- endif %}
    {{- "\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>" }}
    {%- for tool in tools %}
        {{- "\\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\"name\\": <function-name>, \\"arguments\\": <args-json-object>}\\n</tool_call><|im_end|>\\n" }}\n{%- else %}
    {%- if messages[0][\'role\'] == \'system\' %}
        {{- \'<|im_start|>system\\n\' + messages[0][\'content\'] + \'<|im_end|>\\n\' }}
    {%- else %}
        {{- "<|im_start|>system\\nYou are an coding assistant that helps merge code updates. Merge all changes from the <update> snippet into the <code> below, ensuring every modification is fully integrated.
- Preserve the code's structure, order, comments, and indentation exactly.
- Output only the updated code, enclosed within <updated-code> and </updated-code> tags.
- Do not include any additional text, explanations, placeholders, ellipses, or code fences.<|im_end|>\\n" }}
    {%- endif %}\n{%- endif %}\n{%- for message in messages %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) or (message.role == "assistant" and not message.tool_calls) %}
        {{- \'<|im_start|>\' + message.role + \'\\n\' + message.content + \'<|im_end|>\' + \'\\n\' }}
    {%- elif message.role == "assistant" %}
        {{- \'<|im_start|>\' + message.role }}
        {%- if message.content %}
            {{- \'\\n\' + message.content }}
        {%- endif %}
        {%- for tool_call in message.tool_calls %}
            {%- if tool_call.function is defined %}
                {%- set tool_call = tool_call.function %}
            {%- endif %}
            {{- \'\\n<tool_call>\\n{"name": "\' }}
            {{- tool_call.name }}
            {{- \'", "arguments": \' }}
            {{- tool_call.arguments | tojson }}
            {{- \'}\\n</tool_call>\' }}
        {%- endfor %}
        {{- \'<|im_end|>\\n\' }}
    {%- elif message.role == "tool" %}
        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != "tool") %}            {{- \'<|im_start|>user\' }}
        {%- endif %}
        {{- \'\\n<tool_response>\\n\' }}
        {{- message.content }}
        {{- \'\\n</tool_response>\' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- \'<|im_end|>\\n\' }}
        {%- endif %}
    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}
    {{- \'<|im_start|>assistant\\n\' }}
{%- endif %}
"""
