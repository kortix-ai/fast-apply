# v12
template = """Merge all changes from the update snippet to the code below, ensuring that every modification is fully integrated. 
Maintain the code's structure, order, comments, and indentation precisely. 
Do not use any placeholders, ellipses, or omit any sections in <updated-code>.
Only output the updated code; do not include any additional text, explanations, or fences.
\n
<update>{update_snippet}</update>
\n
<code>{original_code}</code>
\n
The updated code MUST be enclosed in <updated-code> tags.
Here's the updated-code with fully integrated changes, start the tag now:
"""