# Fast Apply: Pipeline for Data Generation & Fine-Tuning Qwen2.5 Coder Models

Kortix Fast Apply are models dedicated to perform instant apply with produce full file edit.

To understand what it does, here's the prompt for inference : 

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

Model output :
```
<|im_start|>assistant
<updated-code>[Full-complete updated file]</updatedcode>
```

We choose small model like 7B and 1.5B for fast inference speed, which adapts to the task of instant apply. 


## Data Generation Pipeline

Generating high-quality synthetic data is crucial for training robust models. To adapt with our use case, we choose open-source nextjs-like projects to serve as `original-code`, then use `Claude Sonnet 3.5` (70%) and `GPT-4o` (30%) to generate `update-snippet` and `final-updated-code`.

The pipeline involves the following steps:

1. **Clone Open-Source Repositories**
   ```bash
   git clone --depth 1 https://github.com/your/repo.git data/repo
   ```

2. **Convert Repository Data to Dataset**

   Use `repo_to_dataset.py` to transform the repository data into a structured dataset:

   ```bash
   python data_generation/repo_to_dataset.py /path/to/your/repo \
       --sample-lt-100 50 \
       --sample-100-399 1000 \
       --sample-400-999 800 \
       --sample-1000-1999 600 \
       --sample-2000-2999 400 \
       --sample-3000-3999 200 \
       --sample-4000-4999 100 \
       --sample-5000-9999 50 \
       --sample-10000-plus 25 \
       --output output.parquet \
       --log repo_to_dataset.log \
       --skip 500 \
       --debug
   ```

   **Parameters:**

   - `--sample-lt-100`: Number of samples with fewer than 100 tokens.
   - `--sample-...`
   - `--output`: Output Parquet file name (default: `output.parquet`).
   - `--log`: Log file name (default: `repo.log`).
   - `--skip`: Skip files with fewer than N tokens (default: 0).
   - `--debug`: Enable debug mode for additional logging.

3. **Generate Synthetic Data**

   Depending on your preference for using OpenAI or Anthropic APIs, follow the respective pipelines below.

   ### Using OpenAI's API

   The OpenAI pipeline involves preparing batch data, processing batches, and sending batch requests to generate synthetic updates.

   **Step 1: Prepare Batch Data**

   Convert your dataset into batch request files using `prepare_batch_data.py`:

   ```bash
   python data_generation/openai/prepare_batch_data.py -i data/train2/train_cal_com.parquet -o data/train2/batch2/
   ```

   **Step 2: Process Batch Files**

   Process the prepared batch files using `batch_processor.py`:

   ```bash
   python data_generation/openai/batch_processor.py -i data/train2/batch2/ -o data/train2/train_batch_cal_com.parquet
   ```

   **Step 3: Send Batch Requests**

   Send the processed batch requests to OpenAI's Batch API using `send_batch_request.py`:

   ```bash
   python data_generation/openai/send_batch_request.py -bd data/train2/batch2/ -c 5
   ```

   **Example Workflow:**

   ```bash
   # Prepare batch data
   python data_generation/openai/prepare_batch_data.py -i data/train2/train_cal_com.parquet -o data/train2/batch2/

   # Send batch requests
   python data_generation/openai/send_batch_request.py -bd data/train2/batch2/ -c 5

   # Process batch files
   python data_generation/openai/batch_processor.py -i data/train2/batch2/ -o data/train2/train_batch_cal_com.parquet
   ```

   ### Using Anthropic's API

   The Anthropic pipeline utilizes `synthetic_data_generator.py` to generate synthetic updates based on prompts.

   **Generate Synthetic Data:**

   ```bash
   python data_generation/anthropic/synthetic_data_generator.py --parquet_file data/train2/train_cal_com.parquet
   ```

   **Example Workflow:**

   ```bash
   # Generate synthetic data using Anthropic's API
   python data_generation/anthropic/synthetic_data_generator.py --parquet_file data/train2/train_cal_com.parquet
   ```

4. **Merge Parquet Files**

   Combine multiple Parquet files into a single dataset for consistency:

   ```bash
   python utils/merge_parquet.py data/train2/train_*.parquet --output data/train2/train.parquet
   ```

5. **Convert to JSONL** [Optional]

   For compatibility with certain training frameworks, convert the Parquet dataset to JSONL format:

   ```bash
   python utils/parquet_to_jsonl.py data/train2/train.parquet data/train2/train.jsonl
   ```

## Fine-Tuning the Model

Fine-tuning adjusts the pre-trained Llama3 models to better suit specific tasks or datasets.

1. **Launch the Fine-Tuning Notebook**

   Open `Finetune_Fast-Apply-Llama3-1B-Instruct.ipynb` or `Finetune_Fast-Apply-Llama3-8B-Instruct.ipynb` using [RunPod](https://www.runpod.ai/) or your preferred Jupyter environment.

2. **Configure Training Parameters**

   Within the notebook, set the desired hyperparameters, such as learning rate, batch size, and number of epochs.

3. **Start Fine-Tuning**

   Execute the notebook cells to commence training. Monitor the progress and adjust parameters as needed for optimal performance.

## Model Creation

Creating new model instances is streamlined using the `firectl` tool within the `fireworks/create_model.sh` script. This allows you to instantiate models based on different configurations and base models.

### Models

- **Clone lora adapter**:
  ```bash
  git-lfs install
  git clone <HF-URL>
  ```

The `create_model.sh` script includes commands to create various model versions. Below are examples of how to create these models:

- **1B Model Versions**

  ```bash
  # Create 1B-v13
  firectl create model 1b-v13 1B-v13/ --base-model accounts/fireworks/models/llama-v3p2-1b-instruct -a marko-1d84ff
  ```

- **8B Model Versions**

  ```bash
  # Create 8B-v13-2
  firectl create model 8b-v13-2 8B-v13-2/ --base-model accounts/fireworks/models/llama-v3p1-8b-instruct -a marko-1d84ff
  ```

- **Speculation Decoding 8B Model**

  ```bash
  # Create 8B-v13-2-spec3 with speculation decoding
  firectl create model 8b-v13-2-spec3 8B-v13-2/ \
    --base-model accounts/fireworks/models/llama-v3p1-8b-instruct \
    -a marko-1d84ff \
    --default-draft-model accounts/marko-1d84ff/models/1b-v13-2 \
    --default-draft-token-count 5
  ```

### Usage Instructions

1. **Navigate to the Fireworks Directory**

   ```bash
   cd fireworks
   ```

2. **Run the `create_model.sh` Script**

   Open the `create_model.sh` file and uncomment the desired model creation command. Then execute the script:

   ```bash
   bash create_model.sh
   ```

   Alternatively, you can execute individual `firectl` commands directly in the terminal as shown in the examples above.

## Model Deployment

Deploying your fine-tuned models is made efficient with the `fireworks/deploy.sh` script, which utilizes the `firectl` tool to handle deployment tasks.

### Deployment Commands

The `deploy.sh` script contains commands to deploy specific models. Below are examples:

- **Deploy 8B-v12 Model**

  ```bash
  firectl deploy accounts/marko-1d84ff/models/8b-v12 -a marko-1d84ff
  ```

- **Deploy Specific Model Version**

  ```bash
  firectl deploy accounts/marko-1d84ff/models/1b46eacab949440ab93ad70e72df8428
  ```

### Usage Instructions

1. **Navigate to the Fireworks Directory**

   ```bash
   cd fireworks
   ```

2. **Run the `deploy.sh` Script**

   Open the `deploy.sh` file and uncomment the desired deployment command. Then execute the script:

   ```bash
   bash deploy.sh
   ```

   Alternatively, execute individual `firectl` deployment commands directly in the terminal as shown in the examples above.

### Parameters Explained

- `model_path`: The path to the model you wish to deploy.
- `-a`: Account identifier associated with the model deployment.

## Testing the Deployed Model

Ensuring the deployed model performs as expected involves rigorous testing using predefined test sets and evaluation scripts. This repository provides several tools to facilitate comprehensive testing.

### Inference Test Set Runner

The `fireworks_inference_testset_runner.py` script allows you to benchmark the inference capabilities of your deployed models.

**Usage:**

```bash
python tests_evaluate/fireworks/fireworks_inference_testset_runner.py --input_file data/test/testset.parquet --model_name your-model-name --max_tokens 2000 --num_queries 50
```

**Parameters:**

- `--input_file`: Path to the input Parquet or JSON file containing the test set.
- `--model_name`: Name of the model variant to test (e.g., `3B-v12`).
- `--max_tokens`: Maximum number of tokens per query.
- `--num_queries`: Number of queries to execute (optional).

**Example Workflow:**

```bash
python tests_evaluate/fireworks/fireworks_inference_testset_runner.py \
    --input_file data/test/testset.parquet \
    --model_name 3B-v12 \
    --max_tokens 2000 \
    --num_queries 50
```

### Fireworks Inference Test Set Runner

This script is designed to perform inference on a given test set using the Fireworks API asynchronously, allowing for efficient processing of large datasets.

**Usage:**

```bash
python tests_evaluate/fireworks/fireworks_inference_testset_runner.py --input_file path/to/testset.parquet --model_name your-model-name --max_tokens 2000 --num_queries 100
```

**Parameters:**

- `--input_file`: Path to the input Parquet or JSON file.
- `--model_name`: Name of the model to use for inference.
- `--max_tokens`: Maximum tokens per generation.
- `--num_queries`: Number of queries to process (optional).

**Example:**

```bash
python tests_evaluate/fireworks/fireworks_inference_testset_runner.py \
    --input_file data/test/testset.parquet \
    --model_name 8B-v13-2 \
    --max_tokens 1500 \
    --num_queries 75
```

### Fireworks Throughput Tester

The `test_fireworks.py` script evaluates the throughput and performance of your deployed Fireworks models by measuring the number of tokens processed per second.

**Usage:**

```bash
python tests_evaluate/fireworks/test_fireworks.py --model your-model-name
```

**Parameters:**

- `--model`: The model identifier to use for the test (e.g., `8b-v12`).

**Example Workflow:**

```bash
python tests_evaluate/fireworks/test_fireworks.py --model 8b-v12
```

**What It Does:**

- Executes streaming and non-streaming queries against the specified model.
- Measures and prints the throughput in tokens per second.
- Provides insights into the model's performance under different query loads.

### Evaluation Script

The `evaluate.py` script assesses the quality of the generated code by comparing it against the final code using line difference metrics.

**Usage:**

```bash
python tests_evaluate/evaluate.py input_file1.json input_file2.json --output_file results.json -n 100
```

**Parameters:**

- `input_files`: One or more JSON files containing entries with `final_code` and `generated_text`.
- `--output_file`: (Optional) Path to save the diff results in JSON format.
- `-n`: (Optional) Number of examples to process from each input file.

**Example Workflow:**

```bash
python tests_evaluate/evaluate.py \
    tests_evaluate/results1.json \
    tests_evaluate/results2.json \
    --output_file evaluation_results.json \
    -n 500
```

**What It Does:**

- Parses the `generated_text` to extract code snippets enclosed within `<updated-code>` or `<update-code>` tags.
- Compares the extracted generated code with the `final_code` using the unified diff algorithm.
- Calculates the total number of differing lines, added lines, and removed lines.
- Computes the accuracy score based on the number of fully corrected examples (where there are no differences).
- Outputs detailed statistics for each input file, including average and median diffs, average added/removed lines, and accuracy percentage.

**Sample Output:**

```
Statistics for results1.json:
Total entries: 100
Average total diff: 2.35
Median total diff: 2.00
Average added lines: 1.20
Average removed lines: 1.15
Accuracy score: 85.00%
```
---

Happy Coding!