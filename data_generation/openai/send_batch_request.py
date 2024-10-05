"""
This script sends the batch request to OpenAI's Batch API, monitors its status,
and retrieves the results once completed.
"""

from openai import OpenAI
import argparse
import time
import json
import os
import glob
from datetime import datetime

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def log_batch_info(file_id: str, batch_id: str, config: dict, batch_number: int):
    """
    Logs the batch job information including file ID, batch ID, and configuration details.
    
    Args:
        file_id (str): The ID of the uploaded file.
        batch_id (str): The ID of the created batch job.
        config (dict): Configuration details of the batch job.
    """
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"batch_log_{timestamp}.txt")
    
    with open(log_file, "w") as f:
        f.write(f"Batch Number: {batch_number}\n")
        f.write(f"File ID: {file_id}\n")
        f.write(f"Batch ID: {batch_id}\n")
        f.write("Configuration:\n")
        for key, value in config.items():
            f.write(f"  {key}: {value}\n")
    
    print(f"Batch {batch_number} information logged to {log_file}")

def upload_batch_file(file_path: str):
    """
    Uploads the batch input file to OpenAI's Files API.
    
    Args:
        file_path (str): Path to the .jsonl batch input file.
    
    Returns:
        File: The uploaded file object.
    """
    try:
        with open(file_path, "rb") as file:
            response = client.files.create(file=file, purpose="batch")
        print(f"Uploaded batch file. File ID: {response.id}")
        return response
    except Exception as e:
        print(f"Error uploading batch file: {e}")
        return None

def create_batch_job(file_id: str, endpoint: str, completion_window: str = "24h", metadata: dict = None):
    """
    Creates a batch job using the uploaded file.
    
    Args:
        file_id (str): The ID of the uploaded batch input file.
        endpoint (str): The API endpoint to use (e.g., "/v1/chat/completions").
        completion_window (str, optional): Time window for completion (default: "24h").
        metadata (dict, optional): Additional metadata for the batch job.
    
    Returns:
        Batch: The created batch object.
    """
    try:
        batch = client.batches.create(
            input_file_id=file_id,
            endpoint=endpoint,
            completion_window=completion_window,
            metadata=metadata
        )
        print(f"Created batch job. Batch ID: {batch.id}")
        return batch
    except Exception as e:
        print(f"Error creating batch job: {e}")
        return None

def check_batch_status(batch_id: str):
    """
    Checks the status of the batch job.
    
    Args:
        batch_id (str): The ID of the batch job.
    
    Returns:
        Batch: The retrieved batch object.
    """
    try:
        batch = client.batches.retrieve(batch_id)
        print(f"Batch Status: {batch.status}")
        return batch
    except Exception as e:
        print(f"Error checking batch status: {e}")
        return None

def wait_for_completion(batch_id: str, check_interval: int = 60):
    """
    Waits for the batch job to complete by periodically checking its status.
    
    Args:
        batch_id (str): The ID of the batch job.
        check_interval (int, optional): Seconds between status checks (default: 60).
    
    Returns:
        Batch: The final batch object.
    """
    while True:
        batch = check_batch_status(batch_id)
        if not batch:
            print("Failed to retrieve batch status. Exiting.")
            return None
        if batch.status in ["completed", "failed", "expired", "cancelled"]:
            break
        print(f"Batch job {batch_id} is {batch.status}. Waiting for {check_interval} seconds...")
        time.sleep(check_interval)
    print(f"Batch job {batch_id} has finished with status: {batch.status}")
    return batch

def retrieve_results(batch: object, output_file: str):
    """
    Retrieves the results of the completed batch job and saves them to a file.
    
    Args:
        batch (Batch): The completed batch object.
        output_file (str): Path to save the results.
    """
    if not batch.output_file_id:
        print("No output file ID found. The batch might have failed.")
        return

    try:
        response = client.files.content(batch.output_file_id)
        content = response.read().decode('utf-8')
        
        with open(output_file, 'w') as f:
            f.write(content)

        print(f"Batch results saved to {output_file}")
    except Exception as e:
        print(f"Error retrieving results: {e}")

def process_multiple_batches(batch_dir: str, output_dir: str = None, endpoint: str = "/v1/chat/completions", metadata: dict = None, check_interval: int = 60):
    """
    Process multiple batch files in a directory sequentially.

    Args:
        batch_dir (str): Directory containing batch input files.
        output_dir (str, optional): Directory to save output files. Defaults to batch_dir if not provided.
        endpoint (str, optional): API endpoint to use. Defaults to "/v1/chat/completions".
        metadata (dict, optional): Additional metadata for the batch job.
        check_interval (int, optional): Seconds between status checks. Defaults to 60.
    """
    if output_dir is None:
        output_dir = batch_dir

    batch_files = sorted(glob.glob(os.path.join(batch_dir, "batch_*.jsonl")))
    
    for batch_number, batch_file in enumerate(batch_files, start=1):
        print(f"Processing Batch {batch_number}: {batch_file}")
        
        # Generate output file name
        output_file = os.path.join(output_dir, f"output_batch_{batch_number:03d}.jsonl")
        
        # Upload the batch input file
        batch_file_obj = upload_batch_file(batch_file)
        if not batch_file_obj:
            print(f"Skipping Batch {batch_number} due to error in uploading batch file.")
            continue

        # Create the batch job
        batch = create_batch_job(batch_file_obj.id, endpoint, metadata=metadata)
        if not batch:
            print(f"Skipping Batch {batch_number} due to error in creating batch job.")
            continue

        # Log the batch information
        config = {
            "batch_input_file": batch_file,
            "output_file": output_file,
            "endpoint": endpoint,
            "metadata": metadata,
            "check_interval": check_interval
        }
        log_batch_info(batch_file_obj.id, batch.id, config, batch_number)

        # Wait for the batch job to complete
        completed_batch = wait_for_completion(batch.id, check_interval)

        if completed_batch and completed_batch.status == "completed":
            # Retrieve and save the results
            retrieve_results(completed_batch, output_file)
        else:
            print(f"Batch {batch_number} ended with status: {completed_batch.status if completed_batch else 'Unknown'}. No results to retrieve.")

        print(f"Finished processing Batch {batch_number}\n")

def main():
    parser = argparse.ArgumentParser(description="Send batch requests to OpenAI's Batch API.")
    parser.add_argument("-bd", "--batch_dir", type=str, required=True, help="Directory containing batch input files")
    parser.add_argument("-od", "--output_dir", type=str, help="Directory to save output files (optional, defaults to batch_dir)")
    parser.add_argument("--endpoint", type=str, default="/v1/chat/completions", help="API endpoint to use (default: /v1/chat/completions)")
    parser.add_argument("--metadata", type=str, default=None, help="JSON string of metadata (optional)")
    parser.add_argument("-c", "--check_interval", type=int, default=60, help="Seconds between status checks (default: 60)")

    args = parser.parse_args()

    if not os.path.exists(args.batch_dir):
        print(f"Error: Batch directory {args.batch_dir} does not exist.")
        return

    output_dir = args.output_dir if args.output_dir else args.batch_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if args.metadata:
        try:
            metadata = json.loads(args.metadata)
        except json.JSONDecodeError:
            print("Invalid metadata JSON string. Ignoring metadata.")
            metadata = None
    else:
        metadata = None

    # Ensure API key is set
    if not client.api_key:
        print("Error: OpenAI API key not set. Please set the OPENAI_API_KEY environment variable.")
        return

    # Process all batch files in the directory
    process_multiple_batches(args.batch_dir, output_dir, args.endpoint, metadata, args.check_interval)

if __name__ == "__main__":
    main()

