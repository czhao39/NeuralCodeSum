from __future__ import division

import torch
import warnings
from c2nl.translator import penalties
import itertools
from typing import List

class Beam(object):
    """
    Class for managing the internals of the beam search process.
    Takes care of beams, back pointers, and scores.
    Args:
       size (int): beam size
       pad, bos, eos (int): indices of padding, beginning, and ending.
       n_best (int): nbest size to use
       cuda (bool): use gpu
       global_scorer (:obj:`GlobalScorer`)
    """

    def __init__(self, size, pad, bos, eos,
                 n_best=1, cuda=False,
                 global_scorer=None,
                 min_length=0,
                 stepwise_penalty=False,
                 block_ngram_repeat=0,
                 exclusion_tokens=set()):

        self.size = size # beam size B
        self.tt = torch.cuda if cuda else torch

        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        self.all_scores = []

        # The backpointers at each time-step. (T x B, contains index in size)
        self.prev_ks = []

        # The outputs at each time-step. (T x B, contains word ids)
        self.next_ys = [self.tt.LongTensor(size)
                            .fill_(pad)]
        self.next_ys[0][0] = bos

        # Has EOS topped the beam yet.
        self._eos = eos
        self.eos_top = False

        # The attentions (matrix) for each time.
        self.attn = []

        # Time and k pair for finished.
        self.finished = []
        self.n_best = n_best

        # Information for global scoring.
        self.global_scorer = global_scorer
        self.global_state = {}

        # Minimum prediction length
        self.min_length = min_length

        # Apply Penalty at every step
        self.stepwise_penalty = stepwise_penalty
        self.block_ngram_repeat = block_ngram_repeat
        self.exclusion_tokens = exclusion_tokens

    def get_current_state(self):
        "Get the outputs for the current timestep."
        return self.next_ys[-1]

    def get_current_origin(self):
        "Get the backpointers for the current timestep."
        return self.prev_ks[-1]

    def advance(
      self,
      word_probs,
      attn_out,
      prev_seqs=None,
      dissim=None,
      diversity_weight=0.0,
    ):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attn_out`: Compute and update the beam search.
        Parameters:
        * `word_probs`- probs of advancing from the last step (K x words)
        * `attn_out`- attention at the last step
        * `prev_seqs`: List[List] of shape B'xt, and each item is the index of a word.
        *       Represents a previous hypotheses to compare to for DBS dissimilarity. B' is
        *       the split beam width.
        * `dissim(Beam, List[Beam], num_words)`: Dissimilarity object to compare this Beam
        *       to other Beams
        * `diversity_weight`: weight of dissimilarity function relative to probabilities
        Returns: True if beam search is complete.
        """
        assert(diversity_weight >= 0)
        num_words = word_probs.size(1)
        if self.stepwise_penalty: self.global_scorer.update_score(self, attn_out)
        # force the output to be longer than self.min_length
        cur_len = len(self.next_ys)
        if cur_len < self.min_length:
            for k in range(len(word_probs)):
                word_probs[k][self._eos] = -1e20
        # Sum the previous scores.
        if len(self.prev_ks) > 0:
            beam_scores = word_probs + self.scores.unsqueeze(1).expand_as(word_probs)

            if prev_seqs is not None and len(prev_seqs) > 0:
                assert(len(prev_seqs[0]) == cur_len)
                for k in range(self.size):
                    curr_seq, _ = self.get_hyp(cur_len - 1, k)
                    beam_scores[k, :] += diversity_weight * \
                            dissim.function(prev_seqs, curr_seq, num_words)

            # Don't let EOS have children.
            for i in range(self.next_ys[-1].size(0)):
                if self.next_ys[-1][i] == self._eos: beam_scores[i] = -1e20

            # Block ngram repeats
            if self.block_ngram_repeat > 0:
                le = len(self.next_ys)
                for j in range(self.next_ys[-1].size(0)):
                    hyp, _ = self.get_hyp(le - 1, j)
                    ngrams = set()
                    fail = False
                    gram = []
                    for i in range(le - 1):
                        # Last n tokens, n = block_ngram_repeat
                        gram = (gram + [hyp[i]])[-self.block_ngram_repeat:]
                        # Skip the blocking if it is in the exclusion list
                        if set(gram) & self.exclusion_tokens:
                            continue
                        if tuple(gram) in ngrams:
                            fail = True
                        ngrams.add(tuple(gram))
                    if fail:
                        beam_scores[j] = -1e20
        else:
            beam_scores = word_probs[0]

        flat_beam_scores = beam_scores.view(-1)
        best_scores, best_scores_id = flat_beam_scores.topk(self.size, 0, True, True)

        self.all_scores.append(self.scores)
        self.scores = best_scores

        # best_scores_id is flattened beam x word array, so calculate which
        # word and beam each score came from
        prev_k = best_scores_id // num_words
        self.prev_ks.append(prev_k)
        self.next_ys.append((best_scores_id - prev_k * num_words))
        self.attn.append(attn_out.index_select(0, prev_k))
        self.global_scorer.update_global_state(self)

        for i in range(self.next_ys[-1].size(0)):
            if self.next_ys[-1][i] == self._eos:
                global_scores = self.global_scorer.score(self, self.scores)
                s = global_scores[i]
                self.finished.append((s, len(self.next_ys) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.next_ys[-1][0] == self._eos:
            self.all_scores.append(self.scores)
            self.eos_top = True

    @property
    def done(self):
        return self.eos_top and len(self.finished) >= self.n_best

    def sort_finished(self, minimum=None):
        """
        Return the score and ks=(timestep, k) where each finished sentence ends, sorted by
        sentence length.
        """
        if minimum is not None:
            i = 0
            # Add from beam until we have minimum outputs.
            while len(self.finished) < minimum:
                global_scores = self.global_scorer.score(self, self.scores)
                s = global_scores[i]
                self.finished.append((s, len(self.next_ys) - 1, i))
                i += 1

        self.finished.sort(key=lambda a: -a[0])
        scores = [sc for sc, _, _ in self.finished]
        ks = [(t, k) for _, t, k in self.finished]
        return scores, ks

    def get_hyp(self, timestep, k):
        """
        Walk back to construct the full hypothesis.
        """
        hyp, attn = [], []
        for j in range(len(self.prev_ks[:timestep]) - 1, -1, -1):
            hyp.append(self.next_ys[j + 1][k])
            attn.append(self.attn[j][k])
            k = self.prev_ks[j][k]
        return hyp[::-1], torch.stack(attn[::-1])

class Dissimilarity(object):
    """
    FOR ALL DISSIMILARITY FUNCTIONS:
    Returns: array of size V containing hamming dissimilarity scores for if different words
        were selectd for curr_seq[t]

    Params to each dissimilarity function:
    prev_seqs: array of size B (total beam width) x t (total timesteps so far for
        previous groups), holding the entire hypothesized sequence for each of the previously
        fixed beams.
    curr_seq: array of size t-1 holding the entire hypothesized sequence for the current
        beam, where the element at timestep t will be decided based dissimilarity to prev_seqs
    num_words: vocabulary size V
    Additional kwargs are passed to the dissimilarity function during the init.
    """
    def __init__(self, name, **kwargs):
        assert name in ['hamming', 'cumulative', 'ngram']
        self.name = name

        if name == 'cumulative':
            if 'temperature' not in kwargs.keys(): kwargs['temperature'] = 0.1
        elif name == 'ngram':
            if 'n' not in kwargs.keys(): kwargs['n'] = 2

        for key, value in kwargs.items():
            setattr(self, key, value)
        self.function = getattr(self, name)

    def hamming(self, prev_seqs: ["B=B'xG", "t"], curr_seq: ["t-1"], num_words):
        """
        Penalize selection of tokens used in last time step of previous groups based on
        number of times it was selected for.
        For hamming, we only need the last time step of prev_seqs and curr_seq is technically
        unneeded.
        """
        return -torch.bincount(torch.tensor(prev_seqs)[:,-1], minlength=num_words)

    def cumulative(self, prev_seqs: ["B=B'xG", "t"], curr_seq: ["t-1"], num_words):
        """
        Penalize selection of tokens used at any time step of previous groups, weighted by
        temperature parameter.
        """
        temperature = self.temperature
        T = len(prev_seqs[0])
        count = torch.sum(
                torch.stack([torch.bincount(
                        torch.tensor(prev_seqs)[:,t],
                        weights=torch.eq(torch.tensor(prev_seqs)[:,t], curr_seq[t]),
                        minlength=num_words) for t in range(T-1)]), dim=0)
        return (-count + self.hamming(prev_seqs, curr_seq, num_words))/temperature

    def ngram(self, prev_seqs: ["B=B'xG", "t"], curr_seq: ["t-1"], num_words):
        """
        Penalize selection of token that would create an n-gram that is repeated in
        previous beams by the number of time that n-gram has appeared.
        """
        n = self.n
        ngram = curr_seq[-(n-1):]
        # Count n-grams in prev_seqs
        ngram_counts = [0]*num_words
        for beam in prev_seqs:
            for t in range(len(beam)-(n-1)):
                if beam[t:t+n-1] == ngram: ngram_counts[beam[t+n-1]] += 1
        return -torch.tensor(ngram_counts)


# Diverse beam search, which is essentially just delegating most of the work
# to the existing Beam search module
class DiverseBeam(object):
    def __init__(self, size, pad, bos, eos,
                 n_best=1, cuda=False,
                 global_scorer=None,
                 min_length=0,
                 stepwise_penalty=False,
                 block_ngram_repeat=0,
                 exclusion_tokens=set(),
                 num_groups=1, # the limit where num_groups=1 should be regular beam
                 diversity_weight = 0.0,
                 dissimilarity = 'hamming',
                 **kwargs, # extra args to dissimilarity
    ):
      assert(num_groups > 0)
      assert((size % num_groups) == 0)
      self.split_size = ss = size//num_groups
      self.beams = [Beam(
        ss, pad, bos, eos, n_best, cuda, global_scorer, min_length,
        stepwise_penalty, block_ngram_repeat, exclusion_tokens
      ) for _ in range(num_groups)]
      assert(dissimilarity is not None)
      self.diversity_weight = diversity_weight
      self.dissim = Dissimilarity(dissimilarity, **kwargs)

    # Get the outputs for the current timestep.
    def get_current_state(self):
      return torch.cat([
        beam.get_current_state() for beam in self.beams
      ], dim=0)

    # Get the backpointers for the current timestep.
    def get_current_origin(self):
      return torch.cat([
        beam.get_current_origin() for beam in self.beams
      ], dim=0)

    def advance(self, word_prob, attn_out):
      word_probs = word_prob.split(self.split_size, dim=0)
      attn_outs = attn_out.split(self.split_size, dim=0)
      first_beam = self.beams[0]
      first_beam.advance(
              word_probs[0],
              attn_outs[0],
              prev_seqs=[],
              dissim=self.dissim,
              diversity_weight=self.diversity_weight,
              )
      le = len(first_beam.next_ys)-1
      for i, beam in enumerate(self.beams):
        if i == 0: continue
        prev_seqs = [b.get_hyp(le, k)[0] for k in range(self.split_size) for b in self.beams[:i]]
        beam.advance(
                word_probs[i],
                attn_outs[i],
                prev_seqs=prev_seqs,
                dissim=self.dissim,
                diversity_weight=self.diversity_weight,
                )

    @property
    def done(self): return all(beam.done for beam in self.beams)

    def sort_finished(self, minimum=None):
        all_scores = []
        all_ks = []
        for b, beam in enumerate(self.beams):
            scores, ks = beam.sort_finished(minimum)
            all_scores.append(scores)
            all_ks.append([(t, k + b * self.split_size) for t, k in ks])
        all_scores = list(itertools.chain(*all_scores))
        all_ks = list(itertools.chain(*all_ks))
        resorted = sorted(zip(all_scores, all_ks), key=lambda x: -x[0])
        all_scores, all_ks = [[score for score, k in resorted], [k for score, k in resorted]]
        return all_scores, all_ks

    # Walk back to construct the full hypothesis.
    def get_hyp(self, timestep, k):
        b = k // self.split_size
        kk = k % self.split_size
        hyp, attn = self.beams[b].get_hyp(timestep, kk)
        return hyp, attn


class GNMTGlobalScorer(object):
    """
    NMT re-ranking score from
    "Google's Neural Machine Translation System" :cite:`wu2016google`
    Args:
       alpha (float): length parameter
       beta (float):  coverage parameter
    """

    def __init__(self, alpha, beta, cov_penalty, length_penalty):
        self._validate(alpha, beta, length_penalty, cov_penalty)
        self.alpha = alpha
        self.beta = beta
        penalty_builder = penalties.PenaltyBuilder(cov_penalty,
                                                   length_penalty)
        self.has_cov_pen = penalty_builder.has_cov_pen
        # Term will be subtracted from probability
        self.cov_penalty = penalty_builder.coverage_penalty

        self.has_len_pen = penalty_builder.has_len_pen
        # Probability will be divided by this
        self.length_penalty = penalty_builder.length_penalty

    @classmethod
    def _validate(cls, alpha, beta, length_penalty, coverage_penalty):
        # these warnings indicate that either the alpha/beta
        # forces a penalty to be a no-op, or a penalty is a no-op but
        # the alpha/beta would suggest otherwise.
        if length_penalty is None or length_penalty == "none":
            if alpha != 0:
                warnings.warn("Non-default `alpha` with no length penalty. "
                              "`alpha` has no effect.")
        else:
            # using some length penalty
            if length_penalty == "wu" and alpha == 0.:
                warnings.warn("Using length penalty Wu with alpha==0 "
                              "is equivalent to using length penalty none.")
        if coverage_penalty is None or coverage_penalty == "none":
            if beta != 0:
                warnings.warn("Non-default `beta` with no coverage penalty. "
                              "`beta` has no effect.")
        else:
            # using some coverage penalty
            if beta == 0.:
                warnings.warn("Non-default coverage penalty with beta==0 "
                              "is equivalent to using coverage penalty none.")

    def score(self, beam, logprobs):
        """Rescore a prediction based on penalty functions."""
        len_pen = self.length_penalty(len(beam.next_ys), self.alpha)
        normalized_probs = logprobs / len_pen
        if not beam.stepwise_penalty:
            penalty = self.cov_penalty(beam.global_state["coverage"],
                                       self.beta)
            normalized_probs -= penalty

        return normalized_probs

    def update_score(self, beam, attn):
        """Update scores of a Beam that is not finished."""
        if "prev_penalty" in beam.global_state.keys():
            beam.scores.add_(beam.global_state["prev_penalty"])
            penalty = self.cov_penalty(beam.global_state["coverage"] + attn,
                                       self.beta)
            beam.scores.sub_(penalty)

    def update_global_state(self, beam):
        """Keeps the coverage vector as sum of attentions."""
        if len(beam.prev_ks) == 1:
            beam.global_state["prev_penalty"] = beam.scores.clone().fill_(0.0)
            beam.global_state["coverage"] = beam.attn[-1]
            self.cov_total = beam.attn[-1].sum(1)
        else:
            self.cov_total += torch.min(beam.attn[-1],
                                        beam.global_state['coverage']).sum(1)
            beam.global_state["coverage"] = beam.global_state["coverage"] \
                .index_select(0, beam.prev_ks[-1]).add(beam.attn[-1])

            prev_penalty = self.cov_penalty(beam.global_state["coverage"],
                                            self.beta)
            beam.global_state["prev_penalty"] = prev_penalty
