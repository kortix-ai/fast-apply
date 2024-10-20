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


ONLY_UPDATE_PROMPT = """
You are tasked with generating synthetic data for model training. Your job is to generate an update snippet implementing a meaningful change to the original code based on the specified goal.

**Original code:**
<original_code>
{original_code}
</original_code>

**Specified goal:**
{goal}

**Instructions for creating the update snippet:**

- Focus on implementing the specified goal with precision and detail.
- Keep the snippet concise, including only the new or modified code.
- Example format:
    ```
    // ... existing code ...
    {{ edit_1 }}
    // ... existing code ...
    {{ edit_2 }}
    // ... existing code ...
    ```

- Ensure that the position of the new update is deterministic when applied:
  - Provide enough context to guide the placement of the update by including surrounding lines or clear position markers.
- Do not repeat unchanged parts of the original code unnecessarily.
- Use ellipsis comments (e.g., `// ... existing code ...`, `... existing imports ...`, `... existing logics ...`) only if significant portions are omitted.
- Do not use ellipsis comments if the entire file is updated or replaced.
- Ensure the update is directly related to the specified goal and adheres to any constraints.
- The update snippet must be an exact subset of the final code.
- Write the update snippet within `<update-snippet>` tags.

**Constraints and Guidelines:**

- Be as specific and detailed as possible in your implementation.
- Maintain consistency in coding style, variable and function naming conventions, and overall structure.
- Clearly state any assumptions or context needed for the update.
- Do not include any explanations or comments.

**Begin your response with the update snippet enclosed in `<update-snippet>` tags.**
"""

  # - Alternatively, specify logical placement within the code, e.g., "// Add after function `createContact`".
GOAL = """
Add a new feature: 20
Add two features: 10
Modify an existing function: 25
Modify two existing functions: 15
Delete a feature: 2
Small refactor : 5
Modify a single line: 2
"""

