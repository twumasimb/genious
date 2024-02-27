# import libraries
import os
import sys
import time
import tqdm
import math
import random
import datasets
from datasets import load_dataset, load_from_disk, concatenate_datasets
from transformers import (
    BertConfig,
    BertTokenizerFast,
    BertForPreTraining,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    Trainer, 
    TrainingArguments,
    get_scheduler,
    set_seed,
    SchedulerType)
import torch
from torch.utils.data import DataLoader
import numpy as np
from torch.optim import AdamW
from selectionstrategies import SubmodStrategy
from helper_fns import taylor_softmax_v1

# Set seed
set_seed(42)


# Load dataset
raw_dataset = load_from_disk('./data')

wiki_dataset = load_dataset("wikipedia", "20220301.en")
# Split dataset
wiki_dataset=wiki_dataset["train"].train_test_split(test_size=(30/100), shuffle=False)
wiki_dataset=datasets.DatasetDict({"train": wiki_dataset["train"], "validation": wiki_dataset["test"]})
# saving the dataset
wiki_dataset.save_to_disk("./data")


# Load Tokenizer
print("Creating Tokenizer")
checkpoint = "bert-base-cased"
tokenizer = BertTokenizerFast.from_pretrained(checkpoint)

# Initialize the model 
config = BertConfig()
model = BertForPreTraining(config=config)
model.resize_token_embeddings(len(tokenizer))

# Preprocessing the dataset


# Get the column names for tokenization
column_names = wiki_dataset["train"].column_names
text_column_name = "text" if "text" in column_names else column_names[0]

# set length for tokenization
max_seq_length = tokenizer.model_max_length


# Define a function to tokenize the dataset
def tokenize_function(examples):
    examples[text_column_name] = [
        line for line in examples[text_column_name] if len(line) > 0 and not line.isspace()
    ]
    return tokenizer(examples[text_column_name], truncation=True, max_length=max_seq_length, padding="max_length", return_special_tokens_mask=True)


# Tokenize the dataset
print("Tokenizing the dataset")
tokenized_wiki_dataset = wiki_dataset.map(
    tokenize_function, 
    batched=True, 
    num_proc=4, 
    remove_columns=column_names,
    load_from_cache_file=False,
    desc="Running tokenizer on the entire dataset",
)
train_dataset = tokenized_wiki_dataset["train"]
validation_dataset = tokenized_wiki_dataset["validation"]


tokenized_wiki_dataset.save_to_disk("./tokenized_wiki_data")

# Group the texts


def group_texts(examples, idx, split, tokenized_datasets):
    # Account for [CLS], [SEP], [SEP]
    max_num_tokens = max_seq_length-3
    # We *usually* want to fill up the entire sequence since we are padding
    # to `max_seq_length` anyways, so short sequences are generally wasted
    # computation. However, we *sometimes*
    # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
    # sequences to minimize the mismatch between pre-training and fine-tuning.
    # The `target_seq_length` is just a rough target however, whereas
    # `max_seq_length` is a hard limit.
    short_seq_prob = 0.1
    nsp_probability = 0.5
    target_seq_length = max_num_tokens
    if random.random() < short_seq_prob:
        target_seq_length = random.randint(2, max_num_tokens)
    # We DON'T just concatenate all of the tokens from a document into a long
    # sequence and choose an arbitrary split point because this would make the
    # next sentence prediction task too easy. Instead, we split the input into
    # segments "A" and "B" based on the actual "sentences" provided by the user
    # input.
    result = {k: [] for k, v in tokenizer(
        "", return_special_tokens_mask=True).items()}
    result['next_sentence_label'] = []
    current_chunk = []
    current_length = 0
    i = 0
    while i < len(idx):
        segment = {k: examples[k][i][1:-1] for k in examples.keys()}
        current_chunk.append(segment)
        current_length += len(segment['input_ids'])
        if i == len(idx)-1 or current_length >= target_seq_length:
            if current_chunk:
                # `a_end` is how many segments from `current_chunk` go into the `A`
                # (first) sentence.
                a_end = 1
                if len(current_chunk) >= 2:
                    a_end = random.randint(1, len(current_chunk)-1)
                tokens_a = {k: [] for k, t in tokenizer(
                    "", return_special_tokens_mask=True).items()}
                for j in range(a_end):
                    for k, v in current_chunk[j].items():
                        tokens_a[k].extend(v)

                tokens_b = {k: [] for k, t in tokenizer(
                    "", return_special_tokens_mask=True).items()}
                # Random next
                is_random_next = False
                if len(current_chunk) == 1 or random.random() < nsp_probability:
                    is_random_next = True
                    target_b_length = target_seq_length - \
                        len(tokens_a["input_ids"])
                    # This should rarely go for more than one iteration for large
                    # corpora. However, just to be careful, we try to make sure that
                    # the random document is not the same as the document
                    # we're processing.
                    for _ in range(10):
                        random_segment_index = random.randint(
                            0, len(tokenized_datasets[split])-len(idx)-1)
                        if (random_segment_index-len(idx) not in idx) and (random_segment_index+len(idx) not in idx):
                            break

                    random_start = random.randint(0, len(idx)-1)
                    for j in range(random_start, len(idx)):
                        for k, v in {k: tokenized_datasets[split][random_segment_index+j][k][1:-1] for k in examples.keys()}.items():
                            tokens_b[k].extend(v)
                        if len(tokens_b['input_ids']) >= target_b_length:
                            break
                    # We didn't actually use these segments so we "put them back" so
                    # they don't go to waste.
                    num_unused_segments = len(current_chunk)-a_end
                    i -= num_unused_segments
                # Actual next
                else:
                    is_random_next = False
                    for j in range(a_end, len(current_chunk)):
                        for k, v in current_chunk[j].items():
                            tokens_b[k].extend(v)

                while True:
                    total_length = len(
                        tokens_a['input_ids'])+len(tokens_b['input_ids'])
                    if total_length <= max_num_tokens:
                        break
                    trunc_tokens = tokens_a if len(tokens_a['input_ids']) > len(
                        tokens_b['input_ids']) else tokens_b
                    # We want to sometimes truncate from the front and sometimes from the
                    # back to add more randomness and avoid biases.
                    if random.random() < 0.5:
                        for k in trunc_tokens.keys():
                            del trunc_tokens[k][0]
                    else:
                        for k in trunc_tokens.keys():
                            trunc_tokens[k].pop()
                inp = {
                    k: v[:-1] for k, v in tokenizer("", return_special_tokens_mask=True).items()}
                for k, v in tokens_a.items():
                    inp[k].extend(v)
                SEP = {k: v[1:] for k, v in tokenizer(
                    "", return_special_tokens_mask=True).items()}
                for k, v in SEP.items():
                    inp[k].extend(v)
                tokens_b['token_type_ids'] = list(
                    map(lambda x: 1, tokens_b['token_type_ids']))
                for k, v in SEP.items():
                    tokens_b[k].extend(v)
                tokens_b['token_type_ids'][-1] = 1
                for k, v in tokens_b.items():
                    inp[k].extend(v)
                inp['next_sentence_label'] = int(is_random_next)
                for k, v in inp.items():
                    result[k].append(v)
            current_chunk = []
            current_length = 0
        i += 1
    return result

train_dataset = train_dataset.map(
    group_texts,
    fn_kwargs={'split': 'train', 'tokenized_datasets': tokenized_wiki_dataset},
    batched=True,
    batch_size=1000,
    num_proc=8,
    load_from_cache_file=False,
    with_indices=True,
    desc=f"Grouping Train texts in chunks of {max_seq_length}",
)

train_dataset.save_to_disk("./grouped_data/train")

# Group the validation dataset
validation_dataset = validation_dataset.map(
    group_texts,
    fn_kwargs={'split': 'validation', 'tokenized_datasets': tokenized_wiki_dataset},
    batched=True,
    batch_size=1000,
    num_proc=8,
    load_from_cache_file= False,
    with_indices=True,
    desc=f"Grouping Validation texts in chunks of {max_seq_length}",
)


validation_dataset.save_to_disk("./grouped_data/validation")


# Write a function to load the grouped dataset if it exists. If it does not exist, then group the dataset and save it to disk
train_dataset = load_from_disk("./grouped_data/train")
validation_dataset = load_from_disk("./grouped_data/validation")

# Initialize Random Subset Selection
subset_fraction = 0.1
num_samples = int(round(len(train_dataset) * subset_fraction, 0))
init_subset_indices = [random.sample(list(range(len(train_dataset))), num_samples)]

full_dataset=train_dataset
subset_dataset = full_dataset.select(init_subset_indices[0])

# Data Collator

mlm_probability=0.15
per_device_train_batch_size=32
per_device_eval_batch_size=32

data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=mlm_probability)

# Dataloaders creation
warmstart_dataloader=DataLoader(
    train_dataset, shuffle=True, collate_fn=data_collator, batch_size=per_device_train_batch_size
)

subset_dataloader=DataLoader(
    subset_dataset, shuffle=True, collate_fn=data_collator, batch_size=per_device_train_batch_size
)

eval_dataloader=DataLoader(
    validation_dataset, collate_fn=data_collator, batch_size=per_device_eval_batch_size
)

# Preparing Optimizer & Learning rate schedule

# Optimizer
# Split weights in two groups, one with weight decay and the other not

weight_decay=0.01
learning_rate=5e-5

no_decay=["bias", "LayerNorm.weight"]
optimizer_grouped_parameters=[
    {
        "params":[p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay":weight_decay,
    },
    {
        "params":[p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0
    }
]

optimizer=AdamW(optimizer_grouped_parameters, lr=learning_rate)

lr_scheduler_type=SchedulerType.LINEAR
num_warmup_steps=10
num_training_steps=10

lr_scheduler=get_scheduler(
    name=lr_scheduler_type,
    optimizer=optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)

# Training the Model


num_partitions = 1500
partition_strategy = 'random'
ss_optimizer = 'LazyGreedy'
subset_strategy = SubmodStrategy(logger=None, smi_func_type='fl',
                                 num_partitions=num_partitions, partition_strategy=partition_strategy,
                                 optimizer=ss_optimizer, similarity_criterion='feature',
                                 metric='cosine', eta=1, stopIfZeroGain=False,
                                 stopIfNegativeGain=False, verbose=False, lambdaVal=1)


# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    # del unused_tensor
    torch.cuda.empty_cache()


model.to(device) # Move the model and data to the GPU

max_train_steps = 1000
per_device_train_batch_size = 1
num_warmstart_epochs = 100
num_processes = 1
gradient_accumulation_steps = 1
checkpointing_steps = 1000
output_dir = "./model"

# Train!
total_batch_size = per_device_train_batch_size * num_processes * gradient_accumulation_steps
main_start_time = time.time()
print(f"  Num examples = {len(train_dataset)}")
print(f"  Num warm-start epochs = {num_warmstart_epochs}")
print(f"  Instantaneous batch size per device = {per_device_train_batch_size}")
print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
print(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
print(f"  Total optimization steps = {max_train_steps}")

# Only show the progress bar once on each machine.
# progress_bar = tqdm(range(max_train_steps))
completed_steps = 0

print(f"Begin the training.")
timing = []
warmstart_start_time = time.time()
for epoch in range(num_warmstart_epochs):
    if epoch == 0:
        print("Begin the warm-start")
    model.train()
    for step, batch in enumerate(warmstart_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        start_time = time.time()
        outputs = model(**batch)
        loss = outputs.loss
        print(f"Completed Steps: {1+completed_steps}; Loss: {loss.detach().float()}; lr: {lr_scheduler.get_last_lr()};")
        loss = loss / gradient_accumulation_steps
        loss.backward()
        if step % gradient_accumulation_steps == 0 or step == len(warmstart_dataloader) - 1:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            # progress_bar.update(1)
            completed_steps += 1
        if isinstance(checkpointing_steps, int):
            if completed_steps % checkpointing_steps == 0:
                output_dir = f"step_{completed_steps}"
                if output_dir is not None:
                    output_dir = os.path.join(output_dir, output_dir)
                torch.save(model.state_dict(), output_dir)
        if completed_steps >= max_train_steps:
            break
        timing.append([(time.time() - start_time), 0])

    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs.loss
        losses.append(loss.repeat(per_device_eval_batch_size))

    losses = torch.cat(losses)
    losses = losses[:len(validation_dataset)]
    try:
        perplexity = math.exp(torch.mean(losses))
    except OverflowError:
        perplexity = float("inf")

    print(f"Steps {completed_steps}: perplexity: {perplexity}")
    if epoch == num_warmstart_epochs - 1:
        print("End the warm-start")
# Save the state after warm-start
output_dir = f"after_warmstart_step_{completed_steps}"
if output_dir is not None:
    output_dir = os.path.join(output_dir, output_dir)
torch.save(model.state_dict(), output_dir)
warmstart_end_time = time.time()
print(f"Completed warm-start in {warmstart_end_time - warmstart_start_time} seconds")