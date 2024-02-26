import os
import subprocess
from datetime import datetime
import argparse
from datasets import load_dataset

# I want to test the system with the book corpus
def download_book_corpus(directory):
    # Load the dataset
    dataset_dict = load_dataset('bookcorpus')
    dataset_dict.save_to_disk(directory)

def parse_args():
    parser=argparse.ArgumentParser(description="pre- BERT")
    parser.add_argument(
        "--visible_gpus",
        type=str,
        default="0,1,2,3,4,5,6,7",
        help="visible_gpus"
    )
    parser.add_argument(
        "--main_process_port",
        type=int,
        default=55555,
        help="main process port for accelerate launch"
    )
    args=parser.parse_args()
    return args


def main():
    args=parse_args()
    now=datetime.now()
    timestamp=now.strftime("%d_%m_%Y_%H:%M:%S")
    prefix = "/home/twumasimb/Projects/genious/"
    # Replace '/ingenious-data' with a directory where the user has write permissions, e.g., '/home/user/ingenious-data'
    # Replace '/home/user/ingenious-data' with a directory where the user has write permissions, e.g., '/home/tm0663/ingenious-data'
    if not os.listdir(f'{prefix}datasets'):
        download_book_corpus(f'{prefix}datasets')
    log_dir = prefix + "ingenious-data/logs/bert_uncertainty_sampling_2_"+timestamp+"/"
    model_dir = prefix + "ingenious-data/models/bert_uncertainty_sampling_2_"+timestamp +"/"
    subset_dir = prefix + "ingenious-data/subsets/bert_uncertainty_sampling_2_"+timestamp+"/"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.visible_gpus
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(subset_dir, exist_ok=True)
    l=[
        "accelerate", "launch", "--main_process_port", f"{args.main_process_port}", "mlm_nsp_original.py" #"mlm_nsp_uncertainty_sampling.py", #mlm_nsp_original
        # "--preprocessed",
        "--log_dir", log_dir,
        "--subset_dir", subset_dir,
        "--load_data_from_disk",
        "--data_directory", f"{prefix}/datasets",
        "--tokenizer_name", "bert-base-uncased",
        "--preprocess_batch_size", "1000",
        "--per_device_train_batch_size", "128",
        "--per_device_eval_batch_size", "128",
        "--learning_rate", "1e-4",
        "--lr_max_steps", "1000",
        "--weight_decay" ,"0.01",
        "--max_train_steps", "1000",
        "--gradient_accumulation_steps", "1",
        "--num_warmup_steps", "10000",
        "--output_dir", model_dir,
        # "--seed", "23",
        "--model_type", "bert",
        "--max_seq_length", "128",
        "--preprocessing_num_workers", "8",
        "--mlm_probability" ,"0.15",
        "--short_seq_prob", "0.1",
        "--nsp_probability", "0.5",
        "--subset_fraction", "0.25",
        # "--selection_strategy", "fl",
        # "--optimizer", "LazyGreedy",
        # "--select_every", "25000",
        "--update_losses_every", "125000",
        # "--partition_strategy", "random",
        # "--layer_for_similarity_computation", "9",
        # "--num_partitions", "1500",
        # "--parallel_processes", "48",
        # "--num_warmstart_epochs", "0",
        "--checkpointing_steps", "25000",
        # "--temperature", "1.0",
        # "--resume_from_checkpoint", "/ingenious-data/models/bert_uncertainty_sampling_2__18.43.04/step_500000"
    ]    
    with open(os.path.join(log_dir, "parameters.txt"), "w") as f:
        for i in l:
            f.write(f"{i}\n")
    subprocess.run(l)

if __name__=="__main__":
    main()