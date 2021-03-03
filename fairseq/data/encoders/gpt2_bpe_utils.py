"""
Byte pair encoding utilities from GPT-2.

Original source: https://github.com/openai/gpt-2/blob/master/src/encoder.py
Original license: MIT
"""

from functools import lru_cache
import json
import os
import regex as re
import sys

import pandas as pd


GPT_SPACE_CHAR = '\u0120'


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class Encoder:
    def __init__(self, encoder, bpe_merges, errors='replace',
                 added_tok_fn=os.path.expanduser('~/fairseq/data/gpt2/added_toks.csv')):
        self.encoder = encoder
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = errors  # how to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}
        added_toks_df = pd.read_csv(added_tok_fn)
        added_toks_df['len'] = added_toks_df['tok'].apply(len)
        added_toks_df.sort_values(by='len', ascending=False, inplace=True)
        added_toks = added_toks_df['tok'].tolist()
        added_toks_regex = list(set(list(map(lambda x: ' ?' + re.escape(x.strip(GPT_SPACE_CHAR)), added_toks))))
        sts = '|'.join(added_toks_regex)
        self.pat = re.compile(sts + r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.special_toks_exp = set(added_toks)

    def bpe(self, token):
        if token in self.special_toks_exp:
            return token
        strippped_tok = token.strip(GPT_SPACE_CHAR)
        if strippped_tok in self.special_toks_exp:
            assert strippped_tok.startswith('<')  # is html character
            return strippped_tok
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            for bpe_token in self.bpe(token).split(' '):
                if bpe_token in self.encoder:
                    bpe_tokens.append(self.encoder[bpe_token])
                else:
                    print('Warning. Skipping token --> {}'.format(bpe_token))
        return bpe_tokens

    def decode(self, tokens):
        text = ''.join([self.decoder.get(token, token) for token in tokens])
        start_space = text.startswith(GPT_SPACE_CHAR)
        end_space = text.endswith(GPT_SPACE_CHAR)
        html_adj_text = re.sub(r'Ġ*(</?[dhspe]>)Ġ*', r'Ġ\1Ġ', text)
        # Remove leading and trailing spaces if added by previous line
        if not start_space:
            html_adj_text = html_adj_text.lstrip(GPT_SPACE_CHAR)
        if not end_space:
            html_adj_text = html_adj_text.rstrip(GPT_SPACE_CHAR)
        html_adj_text = re.sub(r'Ġ+', r'Ġ', html_adj_text)
        text = bytearray([self.byte_decoder[c] for c in html_adj_text]).decode('utf-8', errors=self.errors)
        return text


def get_encoder(encoder_json_path, vocab_bpe_path):
    with open(encoder_json_path, "r") as f:
        encoder = json.load(f)
    with open(vocab_bpe_path, "r", encoding="utf-8") as f:
        bpe_data = f.read()
    bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split("\n")[1:-1]]
    return Encoder(
        encoder=encoder,
        bpe_merges=bpe_merges
    )


if __name__ == '__main__':
    tok_dir = os.path.expanduser('~/fairseq/data/gpt2')
    encoder_fn = os.path.join(tok_dir, 'encoder_special_toks.json')
    vocab_fn = os.path.join(tok_dir, 'vocab.bpe')

    bpe_tokenizer = get_encoder(encoder_fn, vocab_fn)
    x = 'phi_month_day_year'
    y = bpe_tokenizer.encode(x)
    print(y)
    z = bpe_tokenizer.decode(y)
    print(z)
    assert x == z
