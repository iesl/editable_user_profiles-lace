"""
Define the loss functions the models might use.
"""
import torch
from torch.nn import functional


class BPRLoss(torch.nn.Module):
    """
    The Bayesian Personalized Ranking loss.

     References
    ----------
    .. [1] Rendle, Steffen, et al. "BPR: Bayesian personalized ranking from
       implicit feedback." Proceedings of the twenty-fifth conference on
       uncertainty in artificial intelligence. AUAI Press, 2009.
    """
    def __init__(self, size_average=False):
        super(BPRLoss, self).__init__()
        self.size_average = size_average

    def forward(self, true_ex_scores, false_ex_scores):
        """
        Push true scores apart from false scores.
        :param true_ex_scores: torch Variable; (batch_size, 1)
        :param false_ex_scores: torch Variable; (batch_size, 1)
        :return: torch Variable.
        """
        diff = true_ex_scores - false_ex_scores
        caseloss = -1.0 * functional.logsigmoid(diff)
        if self.size_average:
            loss = caseloss.mean()
        else:
            loss = caseloss.sum()
        return loss


class NCELoss(torch.nn.Module):
    """
    Noise contrastive estimation like in language modeling/word2vec.

     References
    ----------
    .. https://arxiv.org/pdf/1410.8251.pdf
    .. https://cs224d.stanford.edu/lecture_notes/notes1.pdf
    """
    def __init__(self, size_average=False):
        super(NCELoss, self).__init__()
        self.size_average = size_average

    def forward(self, true_ex_scores, false_ex_scores):
        """
        Push true scores apart from false scores.
        TODO: Allow the presence of more then one false score per true. (k > 1 negative samples.)
        :param true_ex_scores: torch Variable; (batch_size, 1)
        :param false_ex_scores: torch Variable; (batch_size, 1)
        :return: torch Variable.
        """
        pos_ls = torch.nn.functional.logsigmoid(true_ex_scores)
        neg_ls = torch.nn.functional.logsigmoid(-1.0*false_ex_scores)
        caseloss = pos_ls + neg_ls
        if self.size_average:
            loss = caseloss.mean()
        else:
            loss = caseloss.sum()
        # Return the negative of that because we want to minimize the loss.
        return -1.0 * loss


class EntropyLoss(torch.nn.Module):
    """
    Given a set of log probs compute entropy and return a loss for the batch.
    """
    def __init__(self, size_average=False):
        super(EntropyLoss, self).__init__()
        self.size_average = size_average

    def forward(self, log_probs):
        """
        :param log_probs: torch Variable; (batch_size, max_num_targets);
            log probs for each target in each example in the batch. Targets can be
            variable number and the batch is padded with zeros for the max_num_targets.
        :return: torch Variable.
        """
        probs = torch.exp(log_probs)
        entropy = probs * log_probs
        ex_entropy = torch.sum(entropy, dim=1)
        if self.size_average:
            loss = ex_entropy.mean()
        else:
            loss = ex_entropy.sum()
        # Entropy has a negative on the outside. We just want to minimize the entropy.
        return -1.0 * loss


class TrueCrossEntropyLoss(torch.nn.Module):
    """
    Given a set of target probs and log probs from a model compute entropy(target, model)
    and return a loss for the batch.
    """
    def __init__(self, size_average=False):
        super(TrueCrossEntropyLoss, self).__init__()
        self.size_average = size_average

    def forward(self, log_probs, target_log_probs):
        """
        :param log_probs: torch Variable; (batch_size, max_num_targets);
            log probs for each label in each example in the batch. Targets can be
            variable number and the batch is padded with zeros for the max_num_targets.
        :param target_log_probs: torch Variable; (batch_size, max_num_targets);
            target log probability for each label in each example in the batch. Labels can be
            variable number and the batch is padded with zeros for the max_num_targets.
        :return: torch Variable.
        """
        target_probs = torch.exp(target_log_probs)
        centropy = target_probs * log_probs
        ex_cross_entropy = torch.sum(centropy, dim=1)
        if self.size_average:
            loss = ex_cross_entropy.mean()
        else:
            loss = ex_cross_entropy.sum()
        # Cross entropy has a negative on the outside. We just want to minimize the cross entropy.
        return -1.0 * loss
