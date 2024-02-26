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

# set seed
set_seed(42)

# load dataset
dataset = load_from_disk('./data')