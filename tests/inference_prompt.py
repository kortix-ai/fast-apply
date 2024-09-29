#  template="""### Original Code:
#  <code>{original_code}</code>
#
#  ### Modifications:
#  <changes>{update_snippet}</changes>
#
#  ### Task:
#  Apply the changes from <changes> to the <code>, modifying only the specified parts. Keep all other code unchanged. Provide the complete update
#  code as <final_code>.
#
#  ### Final Code:
#  """


## v9
#  template="""You are tasked with updating a code file by applying changes from an update snippet. You will be provided with two inputs:
#
#  1. The original code, enclosed within <original_code> tags:
#  <original_code>{original_code}</original_code>
#
#  2. The update snippet, enclosed within <code_changes> tags:
#  <code_changes>{update_snippet}</code_changes>
#
#  Follow these steps to update the code:
#
#  1. Carefully analyze both the original code and the update snippet.
#  2. Identify the specific parts of the original code that need to be modified based on the update snippet.
#  3. Apply the changes from the update snippet to the appropriate locations in the original code.
#  4. Insert any new additions from the update snippet into logical places within the existing code structure.
#  5. Preserve all unchanged parts of the original code exactly as they appear.
#  6. Maintain the overall structure, formatting, and style of the original code.
#  7. Do not include any comments, markers, or explanations that are not part of the original code or update snippet.
#  8. Ensure that the final output is valid and syntactically correct code in the original language.
#
#  Provide your output in the following format:
#
#  <updated_code>[Your updated code goes here]</updated_code>
#
#  The updated code should be a complete, cohesive piece of code without any additional explanations.
#
#  Now, please provide your updated code based on the original code and update snippet provided earlier."""

## v10
#  template = """### Instruction: Apply the changes from the update snippet to the original code. Output only the updated code, enclosed within <updated_code> tags. Preserve the original formatting and style. Do not include any explanations or additional text outside of the <updated_code> tags.
#
#  ### Original Code:<original_code>{original_code}</original_code>
#
#  ### Code Modification:<changes>{update_snippet}</changes>
#
#  ### Provide your updated code enclosed in <updated_code> tags:"""

#v10.3
#  template = """### Instruction:
#  Apply the changes from the update snippet to the original code. Output only the updated code, enclosed within <updated_code> tags. The updated code must start with <updated_code> and end with </updated_code>. Preserve the original code's structure, order, comments, and indentation. Do not include any explanations or additional text outside of the <updated_code> tags.
#
#  ### Original Code:
#  <original_code>{original_code}</original_code>
#
#  ### Code Modification:
#  <changes>{update_snippet}</changes>
#
#  ### Updated code:
#  """

#v11
#  template = """### Instruction:
#  Apply the modifications provided in the update snippet to the original code.
#  - **Output only the updated code**, enclosed within `<updated_code>` tags.
#  - The updated code **must begin** with `<updated_code>` and **end** with `</updated_code>`.
#  - **Preserve** the original code's structure, order, comments, and indentation.
#  - **Do not include** any explanations or additional text outside of the `<updated_code>` tags.
#
#  ### Original Code:
#  <original_code>{original_code}</original_code>
#
#  ### Code Modifications:
#  <changes>{update_snippet}</changes>
#
#  ### Updated Code:
#  """

# v11-7B
# template = """### Instruction:
# Apply the changes from the update snippet to the original code. Output only the updated code, enclosed within <updated_code> tags. The updated code must start with <updated_code> and end with </updated_code>. Preserve the original code's structure, order, comments, and indentation. Do not include any explanations or additional text outside of the <updated_code> tags.

# ### Original Code:
# <original_code>{original_code}</original_code>

# ### Code Modification:
# <changes>{update_snippet}</changes>

# ### Updated Code:
# """


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