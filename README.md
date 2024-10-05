# Fast Apply: Fine-Tune Llama3 Models

Welcome to **Fast Apply**, a repository dedicated to fine-tuning Llama3 models for enhanced performance and adaptability. This project leverages synthetic data generation, efficient training pipelines, robust testing methodologies, and streamlined model deployment processes to deliver powerful language models tailored to your needs.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Repository Structure](#repository-structure)
4. [Data Generation Pipeline](#data-generation-pipeline)
    - [Using OpenAI's API](#using-openais-api)
    - [Using Anthropic's API](#using-anthropics-api)
5. [Fine-Tuning the Model](#fine-tuning-the-model)
6. [Model Creation](#model-creation)
7. [Model Deployment](#model-deployment)
8. [Testing the Deployed Model](#testing-the-deployed-model)
9. [Contributing](#contributing)
10. [License](#license)

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
├── Finetune_Fast-Apply-Llama3-1B-Instruct.ipynb
├── Finetune_Fast-Apply-Llama3-8B-Instruct.ipynb
├── README.md
├── data_generation/
│   ├── anthropic/
│   ├── openai/
│   └── repo_to_dataset.py
├── download_model.ipynb
├── fireworks/
│   ├── create_model.sh
│   ├── deploy.sh
│   ├── fine-tune/
│   └── deploy.sh
├── html_viewer/
├── requirements.txt
├── tests_evaluate/
└── utils/
```

### Key Files and Directories

- **`Finetune_Fast-Apply-Llama3-1B-Instruct.ipynb`**: Jupyter notebook for fine-tuning the Llama3 1B model.
- **`Finetune_Fast-Apply-Llama3-8B-Instruct.ipynb`**: Jupyter notebook for fine-tuning the Llama3 8B model.
- **`download_model.ipynb`**: Notebook to download the pre-trained model weights.
- **`data_generation/`**: Contains scripts and modules for data generation using OpenAI and Anthropic APIs.
  - **`repo_to_dataset.py`**: Converts repository data into a structured dataset.
  - **Subdirectories**:
    - **`anthropic/`**: Scripts for Anthropic data generation.
    - **`openai/`**: Scripts for OpenAI data generation.
- **`fireworks/`**: Automation scripts for model creation, deployment, and fine-tuning.
  - **`create_model.sh`**: Shell script to create new model instances using `firectl`.
  - **`deploy.sh`**: Shell script to deploy models using `firectl`.
  - **`fine-tune/`**: Scripts related to fine-tuning.
- **`html_viewer/`**: Contains HTML files for viewing diffs and generated JSON data.
- **`requirements.txt`**: Lists all Python dependencies.
- **`tests_evaluate/`**: Contains scripts and modules for evaluating the model.
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

Ensuring the deployed model performs as expected involves rigorous testing using predefined test sets.

1. **Run Serverless vLLM Tests**

   Utilize the `vllm_serverless_tester.py` script to evaluate the model's performance on a serverless infrastructure:

   ```bash
   python tests_evaluate/vllm_runpod/vllm_serverless_tester.py --pod your_pod_id --api_key your_api_key
   ```

2. **Benchmark Inference Test Set**

   Assess the model's inference capabilities with the `inference_testset_runner.py` script:

   ```bash
   python tests_evaluate/inference_testset_runner.py data/test/testset.parquet --pod your_pod_id --api_key your_api_key --model_name 3B-v12 --num_queries 50 --max_tokens 2000
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