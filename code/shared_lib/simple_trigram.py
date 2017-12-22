import numpy as np

from collections import defaultdict


def normalize_counter(c):
    """Given a dictionary of <item, counts>, return <item, fraction>."""
    total = sum(c.itervalues())
    return {w:float(c[w])/total for w in c}


class SimpleTrigramLM(object):
    def __init__(self, words):
        """Build our simple trigram model."""
        # Raw trigram counts over the corpus. 
        # c(w | w_1 w_2) = self.counts[(w_2,w_1)][w]
        self.counts = defaultdict(lambda: defaultdict(lambda: 0.0))
    
        # Iterate through the word stream once.
        w_1, w_2 = None, None
        for word in words:
            if w_1 is not None and w_2 is not None:
                # Increment trigram count.
                self.counts[(w_2,w_1)][word] += 1
            # Shift context along the stream of words.
            w_2 = w_1
            w_1 = word
            
        # Normalize so that for each context we have a valid probability
        # distribution (i.e. adds up to 1.0) of possible next tokens.
        self.probas = defaultdict(lambda: defaultdict(lambda: 0.0))
        for context, ctr in self.counts.iteritems():
            self.probas[context] = normalize_counter(ctr)
            
    def next_word_proba(self, word, seq):
        """Compute p(word | seq)"""
        context = tuple(seq[-2:])  # last two words
        return self.probas[context].get(word, 0.0)
    
    def predict_next(self, seq):
        """Sample a word from the conditional distribution."""
        context = tuple(seq[-2:])  # last two words
        pc = self.probas[context]  # conditional distribution
        words, probs = zip(*pc.iteritems())  # convert to list
        return np.random.choice(words, p=probs)
    
    def score_seq(self, seq, verbose=False):
        """Compute log probability (base 2) of the given sequence."""
        score = 0.0
        count = 0
        # Start at third word, since we need a full context.
        for i in range(2, len(seq)):
            if (seq[i] == "<s>" or seq[i] == "</s>"):
                continue  # Don't count special tokens in score.
            s = np.log2(self.next_word_proba(seq[i], seq[i-2:i]))
            score += s
            count += 1
            # DEBUG
            if verbose:
                print "log P(%s | %s) = %.03f" % (seq[i], " ".join(seq[i-2:i]), s)
        return score, count