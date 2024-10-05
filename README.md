# Fast Apply: Fine-Tune Qwen2.5-Coder

Welcome to **Fast Apply**, a repository dedicated to fine-tuning the Qwen2.5-Coder model for enhanced performance and adaptability. This project leverages synthetic data generation, efficient training pipelines, and robust testing methodologies to deliver a powerful coding assistant tailored to your needs.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Repository Structure](#repository-structure)
4. [Data Generation Pipeline](#data-generation-pipeline)
    - [Using OpenAI's API](#using-openais-api)
    - [Using Anthropic's API](#using-anthropics-api)
5. [Fine-Tuning the Model](#fine-tuning-the-model)
6. [Testing the Deployed Model](#testing-the-deployed-model)
7. [Contributing](#contributing)
8. [License](#license)

## Prerequisites

Before getting started, ensure you have the following installed on your system:

- **Python 3.8+**
- **Git**
- **CUDA-compatible GPU** (for training and inference)
- **[RunPod](https://www.runpod.ai/)** account (for serverless model testing)

Additionally, to utilize the OpenAI and Anthropic data generation pipelines, you need:

- **OpenAI API Key**
- **Anthropic API Key**

Ensure you have these API keys ready and set as environment variables.

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/kortix-ai/fast-apply.git
   cd fast-apply
   ```

2. **Create a Virtual Environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**

   Create a `.env` file in the root directory and add your API keys:

   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ANTHROPIC_API_KEY=your_anthropic_api_key_here
   ```

   Alternatively, you can export them directly in your shell:

   ```bash
   export OPENAI_API_KEY=your_openai_api_key_here
   export ANTHROPIC_API_KEY=your_anthropic_api_key_here
   ```

## Repository Structure

```
.
├── Fast_Apply-Fine-tune-Qwen2.5-Coder-1.5B.ipynb
├── download_model.ipynb
├── html_viewer
│   ├── diff-viewer-debug.html
│   └── generated_data_json-viewer.html
├── repo
│   ├── civitai
│   ├── documenso
│   ├── dub
│   ├── fastapi
│   ├── next.js
│   ├── open-resume
│   ├── papermark
│   ├── photoshot
│   └── refine
├── data_generation
│   ├── repo_to_dataset.py
│   ├── openai
│   │   ├── README.md
│   │   ├── batch_processor.py
│   │   ├── prepare_batch_data.py
│   │   ├── prompt.py
│   │   └── send_batch_request.py
│   └── anthropic
│       ├── prompt_template.py
│       └── synthetic_data_generator.py
├── requirements.txt
├── synthetic_data_generator.py
├── synthetic_data_prompt.py
└── utils
    ├── generate_synthetic_data_prompts_version.xml
    ├── merge_parquet.py
    ├── parquet_to_jsonl.py
    └── print_parquet_columns.py
```

### Key Files and Directories

- **`Fast_Apply-Fine-tune-Qwen2.5-Coder-1.5B.ipynb`**: Jupyter notebook for fine-tuning the Qwen2.5-Coder model.
- **`download_model.ipynb`**: Notebook to download the pre-trained model weights.
- **`html_viewer/`**: Contains HTML files for viewing diffs and generated JSON data.
- **`repo/`**: Submodules and related projects integrated into the main repository.
- **`data_generation/`**: Contains scripts and modules for data generation using OpenAI and Anthropic APIs.
  - **`repo_to_dataset.py`**: Converts repository data into a structured dataset.
  - **`openai/`**: Scripts for generating synthetic data using OpenAI's API.
    - **`batch_processor.py`**: Processes batch files and calculates costs.
    - **`prepare_batch_data.py`**: Prepares batch data for OpenAI's Batch API.
    - **`prompt.py`**: Defines prompt templates and generation functions.
    - **`send_batch_request.py`**: Sends batch requests to OpenAI's Batch API.
  - **`anthropic/`**: Scripts for generating synthetic data using Anthropic's API.
    - **`prompt_template.py`**: Defines prompt templates for Anthropic's API.
    - **`synthetic_data_generator.py`**: Generates synthetic data using Anthropic's API.
- **`requirements.txt`**: Lists all Python dependencies.
- **`utils/`**: Utility scripts for data processing and management.

## Data Generation Pipeline

Generating high-quality synthetic data is crucial for training robust models. The pipeline involves the following steps:

1. **Clone Open-Source Repository**

   ```bash
   git clone --depth 1 https://github.com/your/repo.git data/repo
   ```

2. **Convert Repository Data to Dataset**

   Use `repo_to_dataset.py` to transform the repository data into a structured dataset:

   ```bash
   python data_generation/repo_to_dataset.py /path/to/your/repo --sample-lt-200 20 --sample-200-999 600 --sample-1000-1999 400 --sample-2000-plus 200
   ```

   Copy the output Parquet file to your data folder:

   ```bash
   cp output.parquet data/train2/train_cal_com.parquet
   ```

   **Parameters:**
   - `--sample-lt-200`: Number of samples with fewer than 200 tokens.
   - `--sample-200-999`: Number of samples with 200-999 tokens.
   - `--sample-1000-1999`: Number of samples with 1000-1999 tokens.
   - `--sample-2000-plus`: Number of samples with 2000+ tokens.

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

   # Process batch files
   python data_generation/openai/batch_processor.py -i data/train2/batch2/ -o data/train2/train_batch_cal_com.parquet

   # Send batch requests
   python data_generation/openai/send_batch_request.py -bd data/train2/batch2/ -c 5
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

Fine-tuning adjusts the pre-trained Qwen2.5-Coder model to better suit specific tasks or datasets.

1. **Launch the Fine-Tuning Notebook**

   Open `Fast_Apply-Fine-tune-Qwen2.5-Coder-1.5B.ipynb` using [RunPod](https://www.runpod.ai/) or your preferred Jupyter environment.

2. **Configure Training Parameters**

   Within the notebook, set the desired hyperparameters, such as learning rate, batch size, and number of epochs.

3. **Start Fine-Tuning**

   Execute the notebook cells to commence training. Monitor the progress and adjust parameters as needed for optimal performance.

## Testing the Deployed Model

Ensuring the deployed model performs as expected involves rigorous testing using predefined test sets.

1. **Run Serverless vLLM Tests**

   Utilize the `vllm_serverless_tester.py` script to evaluate the model's performance on a serverless infrastructure:

   ```bash
   python tests/vllm_serverless_tester.py --pod your_pod_id --api_key your_api_key
   ```

2. **Benchmark Inference Test Set**

   Assess the model's inference capabilities with the `inference_testset_runner.py` script:

   ```bash
   python tests/inference_testset_runner.py data/test/testset.parquet --pod your_pod_id --api_key your_api_key --model_name 1.5B-v12 --num_queries 50 --max_tokens 2000
   ```

   **Parameters:**
   - `--pod`: Identifier for the RunPod instance.
   - `--model_name`: Name of the model variant to test.
   - `--num_queries`: Number of first test queries to execute.
   - `--max_tokens`: Maximum tokens per query.

3. **Evaluate Results**

   Review the output logs and metrics to determine the model's accuracy, throughput, and overall performance.

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. **Fork the Repository**

   Click the "Fork" button at the top-right corner of the repository page.

2. **Create a New Branch**

   ```bash
   git checkout -b feature/YourFeatureName
   ```

3. **Commit Your Changes**

   ```bash
   git commit -m "Add your descriptive commit message here"
   ```

4. **Push to the Branch**

   ```bash
   git push origin feature/YourFeatureName
   ```

5. **Create a Pull Request**

   Navigate to the original repository and create a pull request from your forked branch.

## License

This project is licensed under the [MIT License](LICENSE).

---

Happy Coding!