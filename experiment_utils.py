import random

def group_texts(examples, idx, split, tokenizer, max_seq_length, short_seq_prob, nsp_probability, tokenized_datasets):
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
    if random.random()<short_seq_prob:
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
                if len(current_chunk)==1 or random.random()<nsp_probability:
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