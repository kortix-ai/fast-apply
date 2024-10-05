#  PROMPT = """You are an AI assistant tasked with generating synthetic data for model training. Your job is to generate a concise update snippet and the final updated code based on an original code file. Follow these instructions carefully:\n\nFirst, here is the original code:\n<original_code>\n{original_code}\n</original_code>\n\nYour task is to create a concise code snippet that implements an update to the original code. This update snippet should represent a meaningful change or addition to the original code. The update can be one or multiple changes, to add a feature, refactor, or delete code.\n\nWhen creating the update snippet:\n1. Focus on one or more areas of improvement or addition to the original code.\n2. Keep the snippet concise, including only the necessary changes.\n3. Include only the new or modified code in the update snippet, not the entire updated file.\n4. Do not repeat unchanged parts of the original code in the update snippet.\n5. Use ellipsis comments (e.g., \"... rest of the code remains the same ...\") ONLY if there are significant portions of the original code that remain unchanged and are not included in the snippet.\n6. If the entire file is being updated or replaced, do not use any ellipsis comments.\n7. Ensure that the update is relevant to the purpose of generating synthetic data.\n8. The update snippet should be an exact subset of the final code.\n\nWrite your update snippet within <update_snippet> tags.\n\nAfter creating the update snippet, your next task is to provide the complete, updated version of the code file. This final version should incorporate the changes from the update snippet into the original code.\n\nWhen creating the final updated code:\n1. Start with the original code as a base.\n2. Apply the changes from your update snippet.\n3. Ensure that the final version clearly shows how the update is applied to the original code.\n4. Make sure all parts of the code work together cohesively after the update.\n5. Retain all comments from the original code, including those not directly related to the update.\n6. Do not include any ellipsis comments or placeholders in the final code.\n\nWrite the final updated code within <final_code> tags.\n\nThroughout all versions of the code (original, update snippet, and final updated), maintain consistency in:\n1. Coding style (indentation, spacing, etc.)\n2. Variable and function naming conventions\n3. Overall structure and organization of the code\n\nRemember: The update snippet should only contain the new or changed code, while the final code should be the complete updated file.\n\nYour response should begin with the update snippet, followed by the final updated code. Do not include any additional explanation or commentary outside of the specified tags."""

# v10
#  PROMPT = """You are an AI assistant tasked with generating synthetic data for model training. Your job is to generate a concise update snippet and the final updated code based on an original code file. Follow these instructions carefully:\n\nFirst, here is the original code:\n<original_code>\n{original_code}\n</original_code>\n\nYour task is to create a concise code snippet that implements an update to the original code. This update snippet should represent a meaningful change or addition to the original code. The update can be one or multiple changes, to add a feature, refactor, or delete code.\n\nWhen creating the update snippet:\n1. Focus on one or more areas of improvement or addition to the original code.\n2. Keep the snippet concise, including only the necessary changes.\n3. Include only the new or modified code in the update snippet, not the entire updated file.\n4. Do not repeat unchanged parts of the original code in the update snippet.\n5. Use ellipsis comments (e.g., \"... rest of the code remains the same ...\") ONLY if there are significant portions of the original code that remain unchanged and are not included in the snippet.\n6. If the entire file is being updated or replaced, do not use any ellipsis comments.\n7. Ensure that the update is relevant to the purpose of generating synthetic data.\n8. The update snippet should be an exact subset of the final code.\n\nWrite your update snippet within <update_snippet> tags.\n\nAfter creating the update snippet, your next task is to provide the complete, updated version of the code file. This final version should incorporate the changes from the update snippet into the original code.\n\nWhen creating the final updated code:\n1. Start with the original code as a base.\n2. Apply the changes from your update snippet.\n3. Ensure that the final version clearly shows how the update is applied to the original code.\n4. Make sure all parts of the code work together cohesively after the update.\n5. Retain all comments from the original code, including those not directly related to the update.\n6. Do not include any ellipsis comments or placeholders in the final code.\n7. In the final updated code, only make changes based on the <update_snippet>. DO NOT make any additional changes beyond what is specified in the update snippet.\n\nWrite the final updated code within <final_code> tags.\n\nThroughout all versions of the code (original, update snippet, and final updated), maintain consistency in:\n1. Coding style (indentation, spacing, etc.)\n2. Variable and function naming conventions\n3. Overall structure and organization of the code\n\nRemember: The update snippet should only contain the new or changed code, while the final code should be the complete updated file.\n\nYour response should begin with the update snippet, followed by the final updated code. Do not include any additional explanation or commentary outside of the specified tags."""

#V11
# PROMPT = """You are an AI assistant tasked with generating synthetic data for model training. Your job is to generate a concise update snippet and the final updated code based on an original code file. Follow these instructions carefully:\n\nFirst, here is the original code:\n<original_code>\n{original_code}\n</original_code>\n\nYour task is to create a concise code snippet that implements an update to the original code. This update snippet should represent a meaningful change or addition to the original code. The update can be one or multiple changes, to add a feature, refactor, or delete code.\n\nWhen creating the update snippet:\n1. Focus on one or more areas of improvement or addition to the original code.\n2. Keep the snippet concise, including only the necessary changes.\n3. Include only the new or modified code in the update snippet, not the entire updated file.\n4. Do not repeat unchanged parts of the original code in the update snippet.\n5. Use ellipsis comments (e.g., \"... rest of the code remains the same ...\") ONLY if there are significant portions of the original code that remain unchanged and are not included in the snippet.\n6. If the entire file is being updated or replaced, do not use any ellipsis comments.\n7. Ensure that the update is relevant to the purpose of generating synthetic data.\n8. The update snippet should be an exact subset of the final code.\n9.Provide enough context to indicate where the new update should be placed within the original code. This could include surrounding lines or indicators of position (e.g., "... [existing code] ...").\n\nWrite your update snippet within <update_snippet> tags.\n\nAfter creating the update snippet, your next task is to provide the complete, updated version of the code file. This final version should incorporate the changes from the update snippet into the original code.\n\nWhen creating the final updated code:\n1. Start with the original code as a base.\n2. Apply the changes from your update snippet.\n3. Ensure that the final version clearly shows how the update is applied to the original code.\n4. Make sure all parts of the code work together cohesively after the update.\n5. Retain all comments from the original code, including those not directly related to the update.\n6. Do not include any ellipsis comments or placeholders in the final code.\n7. In the final updated code, only make changes based on the <update_snippet>. DO NOT make any additional changes beyond what is specified in the update snippet.\n\nWrite the final updated code within <final_code> tags.\n\nThroughout all versions of the code (original, update snippet, and final updated), maintain consistency in:\n1. Coding style (indentation, spacing, etc.)\n2. Variable and function naming conventions\n3. Overall structure and organization of the code\n\nRemember: The update snippet should only contain the new or changed code, while the final code should be the complete updated file.\n\nYour response should begin with the update snippet, followed by the final updated code. Do not include any additional explanation or commentary outside of the specified tags."""


# V13

PROMPT = """You are tasked with generating synthetic data for model training. Your job is to generate:
1. An update snippet implementing a meaningful change to the original code.
2. The final updated code incorporating this change.

**Original code:**
<original_code>
{original_code}
</original_code>

**Instructions for creating the update snippet:**

- Focus on one or more improvements, additions, or deletions to the original code.
- Keep the snippet concise, including only the new or modified code.
- **Provide enough context to indicate where the update should be placed by including surrounding lines or clear position markers.**
  - Surrounding lines should include **at least one line before and after the updated snippet** to clarify placement.
  - Alternatively, specify **logical placement within the code**, e.g., "add after function `createContact`".
- Do not repeat unchanged parts of the original code unnecessarily.
- Use ellipsis comments (e.g., "... existing code ...") **only if significant portions are omitted**.
- **Do not use ellipsis comments** if the entire file is updated or replaced.
- Ensure the update is directly related to improving, extending, or refining the process of synthetic data generation.
- The update snippet must be an exact subset of the final code.
  
Write the update snippet within `<update_snippet>` tags.

**Instructions for creating the final updated code:**

- Start with the original code.
- **Apply only the changes from the update snippet without including any comments or placement indicators from the update snippet.**
- Ensure the final version clearly shows how the update is applied, without altering unrelated parts of the code.
- Retain all comments from the original code, including those not directly related to the update.
- **Do not include ellipsis comments or placeholders.**
- **Only make changes specified in the `<update_snippet>`**; do not introduce additional changes.

Write the final updated code within `<final_code>` tags.

**Maintain consistency throughout all versions of the code in:**

- Coding style (indentation, spacing, etc.)
- Variable and function naming conventions
- Overall structure and organization

**Remember:**

- The update snippet should only contain the new or changed code, along with context indicators for its placement.
- The final code should be the complete, updated file without any additional comments or placement indicators from the update snippet.
- **Begin your response with the update snippet, followed by the final updated code.**
- Do not include any additional explanations or comments outside the specified tags.

"""
