import os
import torch

def check_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def mask_tokens(inputs: torch.Tensor):
    labels = inputs.clone()
    spec_tokens_id = [0, 1, 2, 3, 4, 5, 6]
    probability_matrix = torch.full(labels.shape, 0.15)

    for spt in spec_tokens_id:
        padding_mask = labels.eq(spt)
        probability_matrix.masked_fill_(padding_mask, value=0.0)

    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = 6
    return inputs, labels

