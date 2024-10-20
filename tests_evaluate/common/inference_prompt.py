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


SYSTEM_PROMPT = """You are an coding assistant that helps merge code updates, ensuring every modification is fully integrated."""

USER_PROMPT = """Merge all changes from the <update> snippet into the <code> below.
- Preserve the code's structure, order, comments, and indentation exactly.
- Output only the updated code, enclosed within <updated-code> and </updated-code> tags.
- Do not include any additional text, explanations, placeholders, ellipses, or code fences.

<code>{original_code}</code>

<update>{update_snippet}</update>

Provide the complete updated code.
"""


chat_template = """
{%- if messages[0]['role'] == 'system' %}
    {{- '<|im_start|>system\n' + messages[0]['content'] + '<|im_end|>\n' }}
{%- else %}
    {{- "<|im_start|>system\nYou are a coding assistant that helps merge code updates, ensuring every modification is fully integrated.<|im_end|>\n" }}
{%- endif %}
{%- for message in messages %}
    {%- if message.role in ["user", "assistant"] or (message.role == "system" and not loop.first) %}
        {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}
"""
