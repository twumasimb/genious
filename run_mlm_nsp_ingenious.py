import argparse
import datetime
import time
import logging
import math
import os
import sys
import random
import datasets
import torch
from torch.optim import AdamW
from datasets import load_dataset, load_from_disk, concatenate_datasets
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import transformers
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed, broadcast_object_list
from transformers import (
    BertConfig,
    BertTokenizerFast,
    BertForPreTraining,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    SchedulerType,
    get_scheduler,
)
from transformers.utils.versions import require_version
from selectionstrategies import SubmodStrategy
from accelerate import InitProcessGroupKwargs
from selectionstrategies.helper_fns import taylor_softmax_v1
import numpy as np
import pickle
import faiss

logger=get_logger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r requirements.txt")

def parse_args():
    parser=argparse.ArgumentParser(description="Train a BERT on Masked Language Modeling and Next Sentence Prediction tasks using the INGENIOUS subset selection strategy")
    parser.add_argument(
        "--log_dir",
        type=str,
        required=True,
        help="The directory to which training logs should be written"
    )
    parser.add_argument(
        "--subset_dir",
        type=str,
        required=True,
        help="The directory to which information regarding selected subsets should be stored"
    )
    parser.add_argument(
        "--preprocessed",
        action="store_true",
        help="If passed, already preprocessed data needs to be given and training will start right away without preprocessing"
    )
    parser.add_argument(
        "--load_data_from_disk",
        action="store_true",
        help="If passed, the dataset is loaded from the disk instead of downloading from the hub"
    )
    parser.add_argument(
        "--data_directory",
        type=str,
        default=None,
        help="The path to the directory containing the dataset in case of load_data_from_disk"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--validation_split_percentage",
        type=float,
        default=5,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--hidden_size", type=int, default=768, help="Hidden Size of language model",
    )
    parser.add_argument(
        "--num_hidden_layers", type=int, default=12, help="Num Hidden Layers",
    )
    parser.add_argument(
        "--num_attention_heads", type=int, default=12, help="Num attention heads",
    )
    parser.add_argument(
        "--intermediate_size", type=int, default=3072, help="Intermediate size",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=30522,
        help="The size of vocabulary used by the tokenizer"
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_max_steps",
        type=int,
        default=1000000,
        help="Max training steps for learning rate. (Can tune the rate of decay of the learning rate with this parameter)"
    )
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay to use.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=250000,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=10000, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=128,
        help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated.",
    )
    parser.add_argument(
        "--line_by_line",
        type=bool,
        default=False,
        help="Whether distinct lines of text in the dataset are to be handled as distinct sequences.",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=96,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--preprocess_batch_size",
        type=int,
        default=None,
        help="batch size during preprocessing"
    )
    parser.add_argument(
        "--overwrite_cache", type=bool, default=False, help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--mlm_probability", type=float, default=0.15, help="Ratio of tokens to mask for masked language modeling loss"
    )
    parser.add_argument(
        "--short_seq_prob", type=float, default=0.1, help="Fraction of input sentences which are not of maximum token length possible"
    )
    parser.add_argument(
        "--nsp_probability", type=float, default=0.5, help="Fraction of incorrect sentence pairs in all of the input"
    )
    parser.add_argument(
        "--subset_fraction", type=float, default=0.25, help="Fraction of the dataset that we want to use for training"
    )
    parser.add_argument(
        "--selection_strategy", type=str, default='fl', help="Subset selection strategy"
    )
    parser.add_argument(
        "--optimizer", type=str, default="LazyGreedy", help="Optimizer to use for subset selection"
    )
    parser.add_argument(
        "--select_every", type=int, default=25000, help="Select a new subset for training every select_every training steps"
    )
    parser.add_argument(
        "--partition_strategy", type=str, default="random", help="Partition strategy for subset selection"
    )
    parser.add_argument(
        "--layer_for_similarity_computation", type=int, default=9, help="The hidden layer to use while calculating the similarities in submodular functions"
    )
    parser.add_argument(
        "--num_partitions", type=int, default=5000, help="Number of partitions in subset selection"
    )
    parser.add_argument(
        "--parallel_processes", type=int, default=96, help="Number of parallel processes for subset selection"
    )
    parser.add_argument(
        "--num_warmstart_epochs", type=int, default=0, help="Number of epochs to run in the warmstart stage"
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether various states should be saved at the end of every n steps",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature while computing the Taylor Softmax"
    )

    args = parser.parse_args()
    return args

def main():
    args=parse_args()
    init_process_group=InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=75000))
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator=Accelerator(kwargs_handlers=[init_process_group])
    # Make one log on every process with the configuration for debugging
    logging.basicConfig(
        filename=os.path.join(args.log_dir,"train_logs.log"),
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    # logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
    
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    if not args.preprocessed:
        logger.info(f"Data is not preprocessed.")
        logger.info(f"Loading the data.")
        if args.load_data_from_disk is not None:
            if args.data_directory is not None:
                raw_datasets=load_from_disk(args.data_directory)
                if "validation" not in raw_datasets.keys():
                    raw_datasets=raw_datasets["train"].train_test_split(test_size=(args.validation_split_percentage/100), shuffle=False)
                    raw_datasets=datasets.DatasetDict({"train": raw_datasets["train"], "validation": raw_datasets["test"]})
        elif args.dataset_name is not None:
            raw_datasets=load_dataset(args.dataset_name, args.dataset_config_name)
            if "validaton" not in raw_datasets.keys():
                raw_datasets=raw_datasets["train"].train_test_split(test_size=(args.validation_split_percentage/100), shuffle=False)
                raw_datasets=datasets.DatasetDict({"train": raw_datasets["train"], "validation": raw_datasets["test"]})
        else:
            data_files={}
            if args.train_file is not None:
                data_files['train']=args.train_file
            if args.validation_file is not None:
                data_files['validation']=args.validation_file
            extension=args.train_file.split(".")[-1]
            if extension=='txt':
                extension='text'
            raw_datasets=load_dataset(extension, data_files=data_files)
            if "validation" not in raw_datasets.keys():
                raw_datasets=raw_datasets["train"].train_test_split(test_size=(args.validation_split_percentage/100), shuffle=False)
                raw_datasets=datasets.DatasetDict({"train": raw_datasets["train"], "validation": raw_datasets["test"]})
    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    logger.info(f"Loading the model configuration.")
    if args.config_name:
        config=BertConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config=BertConfig.from_pretrained(args.model_name_or_path)
    else:
        config=BertConfig(
            vocab_size=args.vocab_size,
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            num_attention_heads=args.num_attention_heads,
            intermediate_size=args.intermediate_size,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=512,
            type_vocab_size=2,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            position_embedding_type="absolute",
        )
    
    logger.info(f"Loading the tokenizer.")
    if args.tokenizer_name:
        tokenizer=BertTokenizerFast.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    elif args.model_name_or_path:
        tokenizer=BertTokenizerFast.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    logger.info(f"Initializing Model.")
    if args.model_name_or_path:
        model=BertForPreTraining.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt"in args.model_name_or_path),
            config=config
        )
    else:
        logger.info("Training a new model from scratch")
        model=BertForPreTraining(config)
    
    model.resize_token_embeddings(len(tokenizer))

    #Preprocessing the datasets
    #First we tokenize all the texts
    if not args.preprocessed:
        column_names=raw_datasets['train'].column_names
        text_column_name="text" if "text" in column_names else column_names[0]
    else:
        column_names=["text"]
        text_column_name="text"
    
    if args.max_seq_length is None:
        max_seq_length=tokenizer.model_max_length
        if max_seq_length>1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length})."
                "Picking 1024 instead. You can change the default value by passing --max_seq_length xxx"
            )
            max_seq_length=1024
    else:
        if args.max_seq_length>tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({args.max_seq_length}) is larger than the maximum length for the "
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length=min(args.max_seq_length, tokenizer.model_max_length)
    
    if not args.preprocessed:
        logger.info(f"Beginning Tokenization.")
        if args.line_by_line:
            #when using line_by_line, we just tokenize each non-empty line.
            padding="max_length" if args.pad_to_max_length else False

            def tokenize_function(examples):
                #remove empty lines
                examples[text_column_name]=[
                    line for line in examples[text_column_name] if len(line)>0 and not line.isspace()
                ]
                return tokenizer(
                    examples[text_column_name],
                    padding=padding,
                    truncation=True,
                    max_length=max_seq_length,
                    #we use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
                    #receives the `special_tokens_mask`
                    return_special_tokens_mask=True
                )
            
            with accelerator.main_process_first():
                tokenized_datasets=raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    num_proc=args.preprocessing_num_workers,
                    remove_columns=[text_column_name],
                    load_from_cache_file=not args.overwrite_cache,
                    desc="Running tokenizer on dataset line by line",
                )
            train_dataset=tokenized_datasets["train"]
            eval_dataset=tokenized_datasets["validation"]
        else:
            # otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
            # We use `return_special_tokens=True` because DataCollatorForLanguageModeling (see below) is more efficient when it
            # receives the `special_tokens_mask`.
            def tokenize_function(examples):
                return tokenizer(examples[text_column_name], return_special_tokens_mask=True)

            with accelerator.main_process_first():
                tokenized_datasets=raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    num_proc=args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not args.overwrite_cache,
                    desc="Running tokenizer on every text in dataset",
                )
            
            #Main data processing function that will concatenate all texts from our dataset and generate chunks of 
            #max_seq_length.
            def group_texts(examples, idx, split):
                # Account for [CLS], [SEP], [SEP]
                max_num_tokens=max_seq_length-3
                # We *usually* want to fill up the entire sequence since we are padding
                # to `max_seq_length` anyways, so short sequences are generally wasted
                # computation. However, we *sometimes*
                # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
                # sequences to minimize the mismatch between pre-training and fine-tuning.
                # The `target_seq_length` is just a rough target however, whereas
                # `max_seq_length` is a hard limit.
                target_seq_length=max_num_tokens
                if random.random()<args.short_seq_prob:
                    target_seq_length=random.randint(2, max_num_tokens)
                # We DON'T just concatenate all of the tokens from a document into a long
                # sequence and choose an arbitrary split point because this would make the
                # next sentence prediction task too easy. Instead, we split the input into
                # segments "A" and "B" based on the actual "sentences" provided by the user
                # input.
                result={k: [] for k, v in tokenizer("", return_special_tokens_mask=True).items()}
                result['next_sentence_label']=[]
                current_chunk=[]
                current_length=0
                i=0 
                while i<len(idx):
                    segment={k: examples[k][i][1:-1] for k in examples.keys()}
                    current_chunk.append(segment)
                    current_length += len(segment['input_ids'])
                    if i==len(idx)-1 or current_length>=target_seq_length:
                        if current_chunk:
                            # `a_end` is how many segments from `current_chunk` go into the `A`
                            # (first) sentence.
                            a_end=1
                            if len(current_chunk)>=2:
                                a_end=random.randint(1, len(current_chunk)-1)
                            tokens_a={k: [] for k, t in tokenizer("", return_special_tokens_mask=True).items()}
                            for j in range(a_end):
                                for k, v in current_chunk[j].items():
                                    tokens_a[k].extend(v)

                            tokens_b={k: [] for k, t in tokenizer("", return_special_tokens_mask=True).items()}
                            # Random next
                            is_random_next=False
                            if len(current_chunk)==1 or random.random()<args.nsp_probability:
                                is_random_next=True
                                target_b_length=target_seq_length-len(tokens_a["input_ids"])
                                # This should rarely go for more than one iteration for large
                                # corpora. However, just to be careful, we try to make sure that
                                # the random document is not the same as the document
                                # we're processing.
                                for _ in range(10):
                                    random_segment_index=random.randint(0, len(tokenized_datasets[split])-len(idx)-1)
                                    if (random_segment_index-len(idx) not in idx) and (random_segment_index+len(idx) not in idx):
                                        break

                                random_start=random.randint(0, len(idx)-1)
                                for j in range(random_start, len(idx)):
                                    for k, v in {k: tokenized_datasets[split][random_segment_index+j][k][1:-1] for k in examples.keys()}.items():
                                        tokens_b[k].extend(v)
                                    if len(tokens_b['input_ids'])>=target_b_length:
                                        break
                                # We didn't actually use these segments so we "put them back" so
                                # they don't go to waste.
                                num_unused_segments=len(current_chunk)-a_end
                                i-=num_unused_segments
                            # Actual next
                            else:
                                is_random_next=False
                                for j in range(a_end, len(current_chunk)):
                                    for k, v in current_chunk[j].items():
                                        tokens_b[k].extend(v)

                            while True:
                                total_length=len(tokens_a['input_ids'])+len(tokens_b['input_ids'])
                                if total_length<=max_num_tokens:
                                    break
                                trunc_tokens= tokens_a if len(tokens_a['input_ids'])>len(tokens_b['input_ids']) else tokens_b
                                # We want to sometimes truncate from the front and sometimes from the
                                # back to add more randomness and avoid biases.
                                if random.random()<0.5:
                                    for k in trunc_tokens.keys():
                                        del trunc_tokens[k][0]
                                else:
                                    for k in trunc_tokens.keys():
                                        trunc_tokens[k].pop()
                            inp={k: v[:-1] for k, v in tokenizer("", return_special_tokens_mask=True).items()}
                            for k, v in tokens_a.items():
                                inp[k].extend(v)
                            SEP={k: v[1:] for k, v in tokenizer("", return_special_tokens_mask=True).items()}
                            for k, v in SEP.items():
                                inp[k].extend(v)
                            tokens_b['token_type_ids']=list(map(lambda x: 1, tokens_b['token_type_ids']))
                            for k, v in SEP.items():
                                tokens_b[k].extend(v)
                            tokens_b['token_type_ids'][-1]=1
                            for k, v in tokens_b.items():
                                inp[k].extend(v)
                            inp['next_sentence_label']=int(is_random_next)
                            for k, v in inp.items():
                                result[k].append(v)
                        current_chunk=[]
                        current_length=0
                    i+=1
                return result
            # Note that with `batched=True`, this map processes 1000 texts together, so group_texts throws away a 
            # remainder for each of those groups of 1000 texts. You can adjust that batch_size here, but a higher value
            # might be slower to preprocess.
            #
            # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map
            train_dataset=tokenized_datasets["train"]
            eval_dataset=tokenized_datasets["validation"]
            logger.info(f"Grouping the tokenized dataset into chunks of {max_seq_length}.")
            with accelerator.main_process_first():
                train_dataset=train_dataset.map(
                    group_texts,
                    fn_kwargs={'split': 'train'},
                    batched=True,
                    batch_size=args.preprocess_batch_size,
                    num_proc=args.preprocessing_num_workers,
                    load_from_cache_file=not args.overwrite_cache,
                    with_indices=True,
                    desc=f"Grouping Train texts in chunks of {max_seq_length}",
                )
            with accelerator.main_process_first():
                eval_dataset=eval_dataset.map(
                    group_texts,
                    fn_kwargs={'split': 'validation'},
                    batched=True,
                    batch_size=args.preprocess_batch_size,
                    num_proc=args.preprocessing_num_workers,
                    load_from_cache_file=not args.overwrite_cache,
                    with_indices=True,
                    desc=f"Grouping Validation texts in chunks of {max_seq_length}",
                )
    else:
        logger.info(f"Loading preprocessed dataset(s) from disk.")
        dataset=load_from_disk(args.data_directory)
        train_dataset=dataset["train"]
        eval_dataset=dataset["validation"]

        nsp_zero=load_from_disk("nsp_zero")
        nsp_one=load_from_disk("nsp_one")
        first_sent_nsp_zero=load_from_disk("first_sent_nsp_zero")
        first_sent_nsp_one=load_from_disk("first_sent_nsp_one")
    #Initial Random Subset Selection
    if accelerator.is_main_process:
        num_samples = int(round(len(train_dataset) * args.subset_fraction, 0))
        init_subset_indices = [random.sample(list(range(len(train_dataset))), num_samples)]
    else:
        init_subset_indices = [[]]
    accelerator.wait_for_everyone()
    broadcast_object_list(init_subset_indices)
    full_dataset=train_dataset
    subset_dataset = full_dataset.select(init_subset_indices[0])

    logger.info(f"Full data has {len(full_dataset)} samples, subset data has {len(subset_dataset)} samples.")
    # Conditional for small test subsets
    if len(train_dataset)>3:
        # Log a few random samples from the training data
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
    
    # Data Collator
    # This one will take care of the randomly masking the tokens.
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=args.mlm_probability)
    data_collator_embd=DataCollatorWithPadding(tokenizer=tokenizer)
    logger.info(f"Creating Data Loaders")
    # Dataloaders creation
    warmstart_dataloader=DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )

    first_sent_nsp_zero_dataloader=DataLoader(
        first_sent_nsp_zero, shuffle=False, collate_fn=data_collator_embd, batch_size=args.per_device_eval_batch_size
    )
    first_sent_nsp_one_dataloader=DataLoader(
        first_sent_nsp_one, shuffle=False, collate_fn=data_collator_embd, batch_size=args.per_device_eval_batch_size
    )

    subset_dataloader=DataLoader(
        subset_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )

    eval_dataloader=DataLoader(
        eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
    )
    logger.info(f"Intializing optimizer, learning rate schedule")
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not
    no_decay=["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters=[
        {
            "params":[p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay":args.weight_decay,
        },
        {
            "params":[p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        }
    ]

    optimizer=AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
    if accelerator.distributed_type==DistributedType.TPU:
        model.tie_weights()

    lr_scheduler=get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.lr_max_steps,
    )

    logger.info(f"Prepare model, optimizer, warmstart_dataloader, first_sent_nsp_zero_dataloader, first_sent_nsp_one_dataloader, subset_dataloader, eval_dataloader with accelerate.")
    # Prepare everything with our `accelerator`
    model, optimizer, warmstart_dataloader, first_sent_nsp_zero_dataloader, first_sent_nsp_one_dataloader, subset_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, warmstart_dataloader, first_sent_nsp_zero_dataloader, first_sent_nsp_one_dataloader, subset_dataloader, eval_dataloader)
    
    if args.selection_strategy in ['fl', 'logdet', 'gc', 'disparity-sum']:
        subset_strategy = SubmodStrategy(logger, args.selection_strategy,
                                    num_partitions=args.num_partitions, partition_strategy=args.partition_strategy,
                                    optimizer=args.optimizer, similarity_criterion='feature', 
                                    metric='cosine', eta=1, stopIfZeroGain=False, 
                                    stopIfNegativeGain=False, verbose=False, lambdaVal=1)
    # Figure out how many steps we should save the Accelerator states
    if hasattr(args.checkpointing_steps, "isdigit"):
        checkpointing_steps=args.checkpointing_steps
        if args.checkpointing_steps.isdigit():
            checkpointing_steps=int(args.checkpointing_steps)
    else:
        checkpointing_steps=None
    
    # Train!
    total_batch_size=args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    main_start_time=time.time()
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num warm-start epochs = {args.num_warmstart_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
        accelerator.load_state(args.resume_from_checkpoint)
    
    logger.info(f"Begin the training.")
    timing = []
    warmstart_start_time=time.time()
    for epoch in range(args.num_warmstart_epochs):
        if epoch==0:
            logger.info("Begin the warm-start")
        model.train()
        for step, batch in enumerate(warmstart_dataloader):
            start_time=time.time()
            outputs=model(**batch)
            loss=outputs.loss
            logger.info(f"Completed Steps: {1+completed_steps}; Loss: {loss.detach().float()}; lr: {lr_scheduler.get_last_lr()};")
            loss=loss/args.gradient_accumulation_steps
            accelerator.backward(loss)
            if step%args.gradient_accumulation_steps==0 or step==len(warmstart_dataloader)-1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps+=1
            if isinstance(checkpointing_steps, int):
                if completed_steps%checkpointing_steps==0:
                    output_dir=f"step_{completed_steps }"
                    if args.output_dir is not None:
                        output_dir=os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)
            if completed_steps>=args.max_train_steps:
                break
            timing.append([(time.time() - start_time), 0])
        
        model.eval()
        losses=[]
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs=model(**batch)

            loss=outputs.loss
            losses.append(accelerator.gather(loss.repeat(args.per_device_eval_batch_size)))

        losses=torch.cat(losses)
        losses=losses[:len(eval_dataset)]
        try:
            perplexity=math.exp(torch.mean(losses))
        except OverflowError:
            perplexity=float("inf")

        logger.info(f"Steps {completed_steps}: perplexity: {perplexity}")
        if epoch==args.num_warmstart_epochs-1:
            logger.info("End the warm-start")
    # Save the state after warm-start
    output_dir=f"after_warmstart_step_{completed_steps}"
    if args.output_dir is not None:
        output_dir=os.path.join(args.output_dir, output_dir)
    accelerator.save_state(output_dir)
    warmstart_end_time=time.time()
    logger.info(f"Completed warm-start in {warmstart_end_time-warmstart_start_time} seconds")

    probs_nsp_zero=[]
    probs_nsp_one=[]
    greedyList_nsp_zero=[]
    greedyList_nsp_one=[]
    gains_nsp_zero=[]
    gains_nsp_one=[]
    if (args.num_warmstart_epochs!=0) or (args.resume_from_checkpoint):
        logger.info(f"Beginning the subset selection after warm-start or resuming from checkpoint")
        start_time=time.time()
        if args.selection_strategy == 'Random-Online':
            if accelerator.is_main_process:
                subset_indices_nsp_zero = [random.sample(list(range(len(first_sent_nsp_zero))), math.floor(num_samples/2))]
                subset_indices_nsp_one = [random.sample(list(range(len(first_sent_nsp_one))), math.ceil(num_samples/2))]
            else:
                subset_indices_nsp_zero = [[]]
                subset_indices_nsp_one = [[]]
        elif args.selection_strategy in ["fl", "logdet", "gc", "disparity-min"]:
            logger.info(f"Performing Subset selection for NSP class 0")
            pbar=tqdm(range(len(first_sent_nsp_zero_dataloader)), disable=not accelerator.is_local_main_process)
            model.eval()
            representations_nsp_zero=[]
            batch_indices_nsp_zero=[]
            total_cnt=0
            total_storage=0

            accelerator.wait_for_everyone()
            unwrapped_model=accelerator.unwrap_model(model)
            bert_model=unwrapped_model.bert
            bert_model=accelerator.prepare(bert_model)
            representations_start_time=time.time()
            for step, batch in enumerate(first_sent_nsp_zero_dataloader):
                with torch.no_grad():
                    output=bert_model(**batch, output_hidden_states=True)
                embeddings=output["hidden_states"][args.layer_for_similarity_computation]
                mask=(batch['attention_mask'].unsqueeze(-1).expand(embeddings.size()).float())
                mask1=((batch['token_type_ids'].unsqueeze(-1).expand(embeddings.size()).float())==0)
                mask=mask*mask1
                mean_pooled=torch.sum(embeddings*mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
                mean_pooled = accelerator.gather(mean_pooled)
                total_cnt += mean_pooled.size(0)
                if accelerator.is_main_process:
                    mean_pooled = mean_pooled.cpu()
                    total_storage += sys.getsizeof(mean_pooled.storage())
                    representations_nsp_zero.append(mean_pooled)
                pbar.update(1)
            if accelerator.is_main_process:
                representations_nsp_zero=torch.cat(representations_nsp_zero, dim=0)
                representations_nsp_zero=representations_nsp_zero[:len(first_sent_nsp_zero)]
                total_storage += sys.getsizeof(representations_nsp_zero.storage())
                representations_nsp_zero=representations_nsp_zero.numpy()
                logger.info('Representations(NSP Class 0) Size: {}, Total number of samples: {}'.format(total_storage/(1024 * 1024), total_cnt))
                batch_indices_nsp_zero=list(range(len(first_sent_nsp_zero)))
                logger.info('Length of indices: {}'.format(len(batch_indices_nsp_zero)))
                logger.info('Representations(NSP Class 0) gathered. Shape of representations: {}. Length of indices: {}'.format(representations_nsp_zero.shape, len(batch_indices_nsp_zero)))
            logger.info(f"Representations(NSP Class 0) computed in {time.time()-representations_start_time} seconds")
            if accelerator.is_main_process:
                partition_indices_nsp_zero, greedyIdx_nsp_zero, gains_nsp_zero = subset_strategy.select(len(batch_indices_nsp_zero)-1, batch_indices_nsp_zero, representations_nsp_zero, parallel_processes=args.parallel_processes, return_gains=True)
                subset_indices_nsp_zero = [[]]
                i=0
                for p in gains_nsp_zero:
                    greedyList_nsp_zero.append(greedyIdx_nsp_zero[i:i+len(p)])         
                    i+=len(p)
                probs_nsp_zero=[taylor_softmax_v1(torch.from_numpy(np.array([partition_gains])/args.temperature)).numpy()[0] for partition_gains in gains_nsp_zero]
                rng=np.random.default_rng(args.seed+completed_steps)
                for i, partition_prob in enumerate(probs_nsp_zero):
                    partition_budget=min(math.ceil((len(partition_prob)/len(batch_indices_nsp_zero)) * math.floor(num_samples/2)), len(partition_prob)-1)
                    subset_indices_nsp_zero[0].extend(rng.choice(greedyList_nsp_zero[i], size=partition_budget, replace=False, p=partition_prob).tolist())
            else:
                subset_indices_nsp_zero=[[]]
        
            logger.info(f"Performing Subset selection for NSP class 1")
            pbar=tqdm(range(len(first_sent_nsp_one_dataloader)), disable=not accelerator.is_local_main_process)
            model.eval()
            representations_nsp_one=[]
            batch_indices_nsp_one=[]
            total_cnt=0
            total_storage=0
            representations_start_time=time.time()
            for step, batch in enumerate(first_sent_nsp_one_dataloader):
                with torch.no_grad():
                    output=bert_model(**batch, output_hidden_states=True)
                embeddings=output["hidden_states"][args.layer_for_similarity_computation]
                mask=(batch['attention_mask'].unsqueeze(-1).expand(embeddings.size()).float())
                mask1=((batch['token_type_ids'].unsqueeze(-1).expand(embeddings.size()).float())==0)
                mask=mask*mask1
                mean_pooled=torch.sum(embeddings*mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
                mean_pooled = accelerator.gather(mean_pooled)
                total_cnt += mean_pooled.size(0)
                if accelerator.is_main_process:
                    mean_pooled = mean_pooled.cpu()
                    total_storage += sys.getsizeof(mean_pooled.storage())
                    representations_nsp_one.append(mean_pooled)
                pbar.update(1)
            if accelerator.is_main_process:
                representations_nsp_one=torch.cat(representations_nsp_one, dim=0)
                representations_nsp_one=representations_nsp_one[:len(first_sent_nsp_one)]
                total_storage += sys.getsizeof(representations_nsp_one.storage())
                representations_nsp_one=representations_nsp_one.numpy()
                logger.info('Representations(NSP Class 1) Size: {}, Total number of samples: {}'.format(total_storage/(1024 * 1024), total_cnt))
                batch_indices_nsp_one=list(range(len(first_sent_nsp_one)))
                logger.info('Length of indices: {}'.format(len(batch_indices_nsp_one)))
                logger.info('Representations(NSP Class 1) gathered. Shape of representations: {}. Length of indices: {}'.format(representations_nsp_one.shape, len(batch_indices_nsp_one)))
            logger.info(f"Representations(NSP Class 1) computed in {time.time()-representations_start_time} seconds")
            if accelerator.is_main_process:
                partition_indices_nsp_one, greedyIdx_nsp_one, gains_nsp_one = subset_strategy.select(len(batch_indices_nsp_one)-1, batch_indices_nsp_one, representations_nsp_one, parallel_processes=args.parallel_processes, return_gains=True)
                subset_indices_nsp_one = [[]]
                i=0
                for p in gains_nsp_one:
                    greedyList_nsp_one.append(greedyIdx_nsp_one[i:i+len(p)])         
                    i+=len(p)
                probs_nsp_one=[taylor_softmax_v1(torch.from_numpy(np.array([partition_gains])/args.temperature)).numpy()[0] for partition_gains in gains_nsp_one]
                rng=np.random.default_rng(args.seed+completed_steps)
                for i, partition_prob in enumerate(probs_nsp_one):
                    partition_budget=min(math.ceil((len(partition_prob)/len(batch_indices_nsp_one)) * math.ceil(num_samples/2)), len(partition_prob)-1)
                    subset_indices_nsp_one[0].extend(rng.choice(greedyList_nsp_one[i], size=partition_budget, replace=False, p=partition_prob).tolist())
            else:
                subset_indices_nsp_one=[[]]
        accelerator.wait_for_everyone()    
        broadcast_object_list(subset_indices_nsp_zero)
        broadcast_object_list(subset_indices_nsp_one)
        timing.append([0, time.time()-start_time])
        logger.info(f"First subset selection completed. Total Time taken(including embeddings computation): {time.time()-start_time}")
    if accelerator.is_main_process:
        output_file=f"nsp_zero_subset_indices_after_step_{completed_steps}.pt"
        output_file=os.path.join(args.subset_dir, output_file)
        torch.save(torch.tensor(subset_indices_nsp_zero), output_file)
        output_file=f"nsp_one_subset_indices_after_step_{completed_steps}.pt"
        output_file=os.path.join(args.subset_dir, output_file)
        torch.save(torch.tensor(subset_indices_nsp_one), output_file)
        output_file=f"nsp_zero_gains_after_step_{completed_steps}.pkl"
        output_file=os.path.join(args.subset_dir, output_file)
        with open(output_file, "wb") as f:
            pickle.dump(gains_nsp_zero, f)
        output_file=f"nsp_one_gains_after_step_{completed_steps}.pkl"
        output_file=os.path.join(args.subset_dir, output_file)
        with open(output_file, "wb") as f:
            pickle.dump(gains_nsp_one, f)
        output_file=f"nsp_zero_partition_indices_after_step_{completed_steps}.pkl"
        output_file=os.path.join(args.subset_dir, output_file)
        with open(output_file, "wb") as f:
            pickle.dump(partition_indices_nsp_zero, f)
        output_file=f"nsp_one_partition_indices_after_step_{completed_steps}.pkl"
        output_file=os.path.join(args.subset_dir, output_file)
        with open(output_file, "wb") as f:
            pickle.dump(partition_indices_nsp_one, f)
        output_file=f"nsp_zero_greedy_indices_after_step_{completed_steps}.pkl"
        output_file=os.path.join(args.subset_dir, output_file)
        with open(output_file, "wb") as f:
            pickle.dump(greedyIdx_nsp_zero, f)
        output_file=f"nsp_one_greedy_indices_after_step_{completed_steps}.pkl"
        output_file=os.path.join(args.subset_dir, output_file)
        with open(output_file, "wb") as f:
            pickle.dump(greedyIdx_nsp_one, f)
    accelerator.wait_for_everyone()

    nsp_zero_subset_dataset=nsp_zero.select(subset_indices_nsp_zero[0])
    nsp_one_subset_dataset=nsp_one.select(subset_indices_nsp_one[0])
    # Concatenate the two datasets
    subset_dataset = concatenate_datasets([nsp_zero_subset_dataset, nsp_one_subset_dataset])
    subset_dataloader=DataLoader(
        subset_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size)
    subset_dataloader = accelerator.prepare(subset_dataloader)

    logger.info("Begin the main training loop with importance re-sampling, after warm-start")
    while completed_steps<args.max_train_steps:
        model.train()
        select_subset=False
        for step, batch in enumerate(subset_dataloader):
            train_time=0
            subset_time=0
            start_time=time.time()
            outputs=model(**batch)
            loss=outputs.loss
            logger.info(f"Completed Steps: {1+completed_steps}; Loss: {loss.detach().float()}; lr: {lr_scheduler.get_last_lr()};")
            loss=loss/args.gradient_accumulation_steps
            accelerator.backward(loss)
            if step%args.gradient_accumulation_steps==0 or step==len(subset_dataloader)-1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps+=1
            train_time += (time.time() - start_time)

            if isinstance(checkpointing_steps, int):
                if completed_steps%checkpointing_steps==0:
                    output_dir=f"step_{completed_steps }"
                    if args.output_dir is not None:
                        output_dir=os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

            if completed_steps>=args.max_train_steps:
                break
            
            if (completed_steps)%args.select_every==0:
                select_subset=True
                break
            timing.append([train_time, subset_time])
        if select_subset==True:
            accelerator.wait_for_everyone()
            start_time = time.time()
            num_samples = int(round(len(full_dataset) * args.subset_fraction, 0)) 
            if args.selection_strategy == 'Random-Online':
                if accelerator.is_main_process:
                    subset_indices_nsp_zero = [random.sample(list(range(len(first_sent_nsp_zero))), math.floor(num_samples/2))]
                    subset_indices_nsp_one = [random.sample(list(range(len(first_sent_nsp_one))), math.ceil(num_samples/2))]
                else:
                    subset_indices_nsp_zero = [[]]
                    subset_indices_nsp_one = [[]]
            elif args.selection_strategy in ["fl", "logdet", "gc", "disparity-min"]:
                logger.info(f"Performing Subset selection for NSP class 0")
                sampling_start_time=time.time()
                if accelerator.is_main_process:
                    subset_indices_nsp_zero=[[]]
                    rng=np.random.default_rng(args.seed+completed_steps)
                    for i, partition_prob in enumerate(probs_nsp_zero):
                        partition_budget=min(math.ceil((len(partition_prob)/len(batch_indices_nsp_zero)) * math.floor(num_samples/2)), len(partition_prob)-1)
                        subset_indices_nsp_zero[0].extend(rng.choice(greedyList_nsp_zero[i], size=partition_budget, replace=False, p=partition_prob).tolist())
                else:
                    subset_indices_nsp_zero=[[]]
                logger.info("Sampling time(NSP Class 0): {}".format(time.time()-sampling_start_time))
            
                logger.info(f"Performing Subset selection for NSP class 1")
                sampling_start_time=time.time()
                if accelerator.is_main_process:
                    subset_indices_nsp_one=[[]]
                    rng=np.random.default_rng(args.seed+completed_steps)
                    for i, partition_prob in enumerate(probs_nsp_one):
                        partition_budget=min(math.ceil((len(partition_prob)/len(batch_indices_nsp_one)) * math.ceil(num_samples/2)), len(partition_prob)-1)
                        subset_indices_nsp_one[0].extend(rng.choice(greedyList_nsp_one[i], size=partition_budget, replace=False, p=partition_prob).tolist())
                else:
                    subset_indices_nsp_one=[[]]
                logger.info("Sampling time(NSP Class 1): {}".format(time.time()-sampling_start_time))
            accelerator.wait_for_everyone()
            broadcast_object_list(subset_indices_nsp_zero)
            broadcast_object_list(subset_indices_nsp_one)
            timing.append([0, time.time()-start_time])
            if accelerator.is_main_process:
                output_file=f"nsp_zero_subset_indices_after_step_{completed_steps}.pt"
                output_file=os.path.join(args.subset_dir, output_file)
                torch.save(torch.tensor(subset_indices_nsp_zero), output_file)
                output_file=f"nsp_one_subset_indices_after_step_{completed_steps}.pt"
                output_file=os.path.join(args.subset_dir, output_file)
                torch.save(torch.tensor(subset_indices_nsp_one), output_file)
            accelerator.wait_for_everyone()

            nsp_zero_subset_dataset=nsp_zero.select(subset_indices_nsp_zero[0])
            nsp_one_subset_dataset=nsp_one.select(subset_indices_nsp_one[0])
            # Concatenate the two datasets
            subset_dataset = concatenate_datasets([nsp_zero_subset_dataset, nsp_one_subset_dataset])
            subset_dataloader=DataLoader(
                subset_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size)
            subset_dataloader = accelerator.prepare(subset_dataloader)

            subset_time=time.time()-start_time
            select_subset=False
            timing.append([0, subset_time])
            logger.info(f"Subset selection time(total resampling time): {time.time()-start_time} seconds.")
        model.eval()
        losses=[]
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs=model(**batch)

            loss=outputs.loss
            losses.append(accelerator.gather(loss.repeat(args.per_device_eval_batch_size)))

        losses=torch.cat(losses)
        losses=losses[:len(eval_dataset)]
        try:
            perplexity=math.exp(torch.mean(losses))
        except OverflowError:
            perplexity=float("inf")

        logger.info(f"Steps {completed_steps}: perplexity: {perplexity}")
    
    logger.info(f"Timing: {timing}")
    logger.info(f"Saving the final model after {completed_steps} steps.")
    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model=accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
    logger.info(f"Training completed successfully in {time.time()-main_start_time} seconds.")

if __name__ == "__main__":
    main()