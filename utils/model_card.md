---
base_model: unsloth/qwen2.5-coder-7b-instruct-bnb-4bit
language:
- en
license: apache-2.0
tags:
- text-generation-inference
- transformers
- unsloth
- qwen2
- trl
- sft
- fast-apply
- instant-apply
---


# FastApply-7B-v1.0

[Github: kortix-ai/fast-apply](https://github.com/kortix-ai/fast-apply)   
[Dataset: Kortix/FastApply-dataset-v1.0](https://huggingface.co/datasets/Kortix/FastApply-dataset-v1.0)

## Model Details

### Basic Information

- **Developed by:** Kortix
- **License:** apache-2.0
- **Finetuned from model:** [unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit](https://huggingface.co/unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit)

### Model Description

FastApply-7B-v1.0 is a 7B model designed for instant code application, producing full file edits to power [SoftGen AI](https://softgen.ai/).    
It is part of the Fast Apply pipeline for data generation and fine-tuning Qwen2.5 Coder models.

The model achieves high throughput when deployed on fast providers like Fireworks while maintaining high edit accuracy, with a speed of approximately 150 tokens/second.

## Intended Use

FastApply-7B-v1.0 is intended for use in AI-powered code editors and tools that require fast, accurate code modifications. It is particularly well-suited for:

- Instant code application tasks
- Full file edits
- Integration with AI-powered code editors like Aider and PearAI
- Local tools to reduce the cost of frontier model output

## Inference template

FastApply-7B-v1.0 is based on the Qwen2.5 Coder architecture and is fine-tuned for code editing tasks. It uses a specific prompt structure for inference:

```
<|im_start|>user
Merge all changes from the <update> snippet into the <code> below.
- Preserve the code's structure, order, comments, and indentation exactly.
- Output only the updated code, enclosed within <updated-code> and </updated-code> tags.
- Do not include any additional text, explanations, placeholders, ellipses, or code fences.

<code>{original_code}</code>

<update>{update_snippet}</update>

Provide the complete updated code."""
```

The model's output is structured as:

```
<|im_start|>assistant
<updated-code>[Full-complete updated file]</updated-code>
```

## Additional Information

For more details on the Fast Apply pipeline, data generation process, and deployment instructions, please refer to the [GitHub repository](https://github.com/Kortex/FastApply).

## How to Use

To use the model, you can load it using the Hugging Face Transformers library:


```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Kortix/FastApply-7B-v1.0")
tokenizer = AutoTokenizer.from_pretrained("Kortix/FastApply-7B-v1.0")

# Prepare your input following the prompt structure mentioned above
input_text = """<|im_start|>system
You are a coding assistant that helps merge code updates, ensuring every modification is fully integrated.<|im_end|>
<|im_start|>user
Merge all changes from the <update> snippet into the <code> below.
- Preserve the code's structure, order, comments, and indentation exactly.
- Output only the updated code, enclosed within <updated-code> and </updated-code> tags.
- Do not include any additional text, explanations, placeholders, ellipses, or code fences.

<code>{original_code}</code>

<update>{update_snippet}</update>

Provide the complete updated code.<|im_end|>
<|im_start|>assistant
"""

input_text = input_text.format(
    original_code=original_code,
    update_snippet=update_snippet,
).strip() + tokenizer.eos_token 

# Generate the response
input_ids = tokenizer.encode(input_text, return_tensors="pt")
output = model.generate(input_ids, max_length=8192)
response = tokenizer.decode(output[0])

# Extract the updated code from the response
updated_code = response.split("<updated-code>")[1].split("</updated-code>")[0]

print(updated_code)
```