# Fast Apply: Fine-Tune Qwen2.5-Coder

Welcome to **Fast Apply**, a repository dedicated to fine-tuning the Qwen2.5-Coder model for enhanced performance and adaptability. This project leverages synthetic data generation, efficient training pipelines, and robust testing methodologies to deliver a powerful coding assistant tailored to your needs.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Repository Structure](#repository-structure)
4. [Data Generation Pipeline](#data-generation-pipeline)
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
├── repo_to_dataset.py
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
- **`repo_to_dataset.py`**: Script to convert repository data into a structured dataset.
- **`synthetic_data_generator.py`**: Generates synthetic data based on prompts.
- **`synthetic_data_prompt.py`**: Contains prompt templates for data generation.
- **`utils/`**: Utility scripts for data processing and management.

## Data Generation Pipeline

Generating high-quality synthetic data is crucial for training robust models. The pipeline involves the following steps:

1. **Clone open-source repo**

   ```bash
   git clone --depth 1 new/repo/as/training/data
   ```

2. **Convert Repository Data to Dataset**

   Use `repo_to_dataset.py` to transform the repository data into a structured dataset:

   ```bash
   python repo_to_dataset.py /path/to/your/repo --sample-lt-200 20 --sample-200-999 600 --sample-1000-1999 400
   ```

   ```bash
   cp output.parquet to/your/data/folder
   ```

   **Parameters:**
   - `--sample-lt-200`: Number of samples with fewer than 200 tokens.
   - `--sample-200-999`: Number of samples with 200-999 tokens.
   - `--sample-1000-1999`: Number of samples with 1000-1999 tokens.
   - ...

3. **Generate Synthetic Data**

   Run the synthetic data generator to create new training examples:

   ```bash
   python synthetic_data_generator.py --parquet_file data/train_[name_data_file].parquet
   ```

   This script utilizes predefined prompts to generate updated code snippets and final code versions, enhancing the dataset's diversity and quality.

4. **Merge Parquet Files**

   Combine multiple Parquet files into a single dataset for consistency:

   ```bash
   python utils/merge_parquet.py data/train_*.parquet --output data/train.parquet
   ```

5. **Convert to JSONL** [Optional]

   For compatibility with certain training frameworks, convert the Parquet dataset to JSONL format:

   ```bash
   python utils/parquet_to_jsonl.py data/train.parquet data/train.jsonl
   ```

## Fine-Tuning the Model

Fine-tuning adjusts the pre-trained Qwen2.5-Coder model to better suit specific tasks or datasets.

1. **Launch the Fine-Tuning Notebook**

   Open `Fast_Apply-Fine-tune-Qwen2.5-Coder-1.5B.ipynb` using `Runpod`.

2. **Configure Training Parameters**

   Within the notebook, set the desired hyperparameters, such as learning rate, batch size, and number of epochs.

3. **Start Fine-Tuning**

   Execute the notebook cells to commence training. Monitor the progress and adjust parameters as needed for optimal performance.

## Testing the Deployed Model

Ensuring the deployed model performs as expected involves rigorous testing using predefined test sets.

1. **Run Serverless vLLM Tests**

   Utilize the ` vllm_serverless_tester.py` script to evaluate the model's performance on a serverless infrastructure:

   ```bash
   python tests/ vllm_serverless_tester.py --pod your_pod_id
   ```

2. **Benchmark Inference Test Set**

   Assess the model's inference capabilities with the `run_inference_testset.py` script:

   ```bash
   python tests/run_inference_testset.py data/test_100.parquet --pod your_pod_id --model_name 1.5B-v12 --num_queries 50
   ```

   **Parameters:**
   - `--pod`: Identifier for the RunPod instance.
   - `--model_name`: Name of the model variant to test.
   - `--num_queries`: Number of first test queries to execute.

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

