from collections import defaultdict
from functools import reduce

word_in_spam = defaultdict(int)
word_in_ham  = defaultdict(int)

def p_word_given_spam(word, n_spam):
    return (word_in_spam[word] + 1) / (n_spam + 2)

def p_word_given_ham(word, n_ham):
    return (word_in_ham[word] + 1) / (n_ham + 2)

# ── Paso 3: P(S|W) por palabra via Bayes ─────────────────────────
def p_spam_given_word(word, P_S, P_H):
    p_ws = p_word_given_spam(word)
    p_wh = p_word_given_ham(word)
    return (p_ws * P_S) / (p_ws * P_S + p_wh * P_H)


