"""
Functions used across models.
"""
import numpy as np
import torch
from torch.nn import functional
from torch.autograd import Variable


def sparse_masking_softmax(x, seq_lens, sparsity):
    """
    Performs masked softmax.
    Masked softmax from: https://gist.github.com/kaniblu/94f3ede72d1651b087a561cf80b306ca
    :param x: [batch_size, num_items]
    :param seq_lens: list(int);
    :param sparsity: float; [0.0, 1.0]; the proportion of values to remove.
    :return:
        sm: softmax with non pruned values used for computing softmax values.
        zero_mask: bool tensor which says which values were masked out. It is
            True wherever we want to mask out a value.
    """
    assert(0 < sparsity <= 1.0)
    bsize, max_seq_len = x.shape[0], x.shape[1]
    prop_to_keep = 1-sparsity
    # for every batch element compute how many NON ZEROS there will be - minimum is 3;
    # this is the index for the threshold - everything less than this is pruned.
    pruned_seq_lens = []
    for sl in seq_lens:
        # todo: because of python float behaviour the pruning will give one fewer; use Decimal instead. --low-pri.
        if sl > 3:  # If there are more than 3 elements keep atleast 3.
            pruned_seq_lens.append(max(int(sl*prop_to_keep), 3)-1)
        else:  # If there are fewer than 3 elements then use the last element - masking keeps >= this value.
            pruned_seq_lens.append(sl-1)
    sparse_prop = Variable(torch.LongTensor(pruned_seq_lens))
    if torch.cuda.is_available():
        sparse_prop = sparse_prop.cuda()
    # Sort the values.
    cm_sorted = torch.sort(x, dim=1, descending=True)[0]
    # Get the set of params which are at the threshold value for the
    # desired level of sparsity.
    threshs = cm_sorted[torch.arange(bsize), sparse_prop[torch.arange(bsize)]]
    # Compute the mask - whatever is greater than or equal to the threshold will be kept.
    mask = torch.ge(x, threshs.unsqueeze(dim=1)).float()
    
    # Perform the masked softmax.
    entire_row_notzero = mask.sum(dim=1) > 0
    # Set the to be masked items to negative infinity.
    x_masked = x * mask + (1 - 1 / mask)
    # If an entire row is zeros then set it to zero.
    x_masked = x_masked * entire_row_notzero.unsqueeze(dim=1).float()
    # Get the max value for numerically stable softmax.
    x_max = x_masked.max(1)[0]
    # Exponentiate.
    x_exp = (x - x_max.unsqueeze(-1)).exp()
    # Mask out the to be masked values.
    neg_mask = (1-mask).bool()
    x_exp = x_exp.masked_fill(neg_mask, 10e-8)
    denominator = x_exp.sum(dim=1) + 10e-8
    # Normalize and return.
    sm = x_exp / denominator.unsqueeze(-1)
    return sm, neg_mask
    

def masked_2d_softmax(batch_scores, target_lens1, target_lens2):
    """
    Given the scores for the assignments for every example in the batch apply
    a masked softmax for the variable number of assignments.
    :param batch_scores: torch Tensor; batch_size x dim1 x dim2; With non target
        scores set to zero.
    :param target_lens1: list(int) [batch_size]; number of elemenets over which to
        compute softmax in each example of the batch along dim 1.
    :param target_lens2: list(int) [batch_size]; number of elemenets over which to
        compute softmax in each example of the batch along dim 2.
    :return: probs: torch Tensor; same size as batch_scores.
    """
    batch_size, q_max_size, c_max_size = batch_scores.size()
    # Set all the logits beyond the targets to very large negative values
    # so they contribute minimally to the softmax.
    logit_mask = np.zeros((batch_size, q_max_size, c_max_size))
    for i, (len1, len2) in enumerate(zip(target_lens1, target_lens2)):
        logit_mask[i, len1:, :] = -10e8
        logit_mask[i, :, len2:] = -10e8
    logit_mask = Variable(torch.FloatTensor(logit_mask))
    if torch.cuda.is_available():
        logit_mask = logit_mask.cuda()
    # Work with log probabilities because its the numerically stable softmax.
    batch_scores = batch_scores + logit_mask
    log_probs = functional.log_softmax(batch_scores.view(batch_size, q_max_size*c_max_size), dim=1)
    log_probs = log_probs.view(batch_size, q_max_size, c_max_size)
    return log_probs.exp()
