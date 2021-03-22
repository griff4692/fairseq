# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

from fairseq import utils
from fairseq.data import (
    encoders,
    AppendTokenDataset,
    DenoisingDataset,
    Dictionary,
    PrependTokenDataset,
    StripTokenDataset,
    TokenBlockDataset,
    data_utils,
)
from fairseq.data.encoders.utils import get_whole_word_mask
from fairseq.data.shorten_dataset import maybe_shorten_dataset


class Args:
    def __init__(self):
        self.gpt2_encoder_json = os.path.expanduser('~/fairseq/data/gpt2/encoder_special_toks.json')
        self.gpt2_vocab_bpe = os.path.expanduser('~/fairseq/data/gpt2/vocab.bpe')
        self.bpe = 'gpt2'

        self.mask = 0.3
        self.permute_sentences = 1.0
        self.mask_length = 'span-poisson'
        self.mask_random = 0.1
        self.insert = 0.0
        self.rotate = 0.0
        self.replace_length = 1
        self.poisson_lambda = 3.0
        self.sample_break_mode = 'eos'


if __name__ == '__main__':
    bin_dir = os.path.expanduser('~/fairseq/data/bin_new')
    vocab_dir = os.path.expanduser('~/fairseq/data/gpt2')
    dictionary = Dictionary.load(os.path.join(vocab_dir, 'dict.txt'))
    source_dictionary = dictionary
    target_dictionary = dictionary
    args = Args()
    mask_length = 'span-poisson'

    bpe = encoders.build_bpe(args)

    mask_idx = dictionary.add_symbol('<mask>')
    seed = 1992
    tokens_per_sample = 512
    dataset_impl = 'mmap'
    shorten_method = None
    shorten_data_split_list = ''

    paths = utils.split_paths(bin_dir)
    assert len(paths) > 0
    split = 'valid'
    data_path = paths[0]
    split_path = os.path.join(data_path, split)

    dataset = data_utils.load_indexed_dataset(
        split_path,
        dictionary,
        dataset_impl,
        combine=False,
    )

    dataset = StripTokenDataset(dataset, dictionary.eos())

    dataset = maybe_shorten_dataset(
        dataset,
        split,
        shorten_data_split_list,
        shorten_method,
        tokens_per_sample,
        seed,
    )

    prev_size = len(dataset)

    # create continuous blocks of tokens
    dataset = TokenBlockDataset(
        dataset,
        dataset.sizes,
        tokens_per_sample - 2,  # one less for <s> and one for </s>
        pad=dictionary.pad(),
        eos=dictionary.eos(),
        break_mode=args.sample_break_mode,
        document_sep_len=0,
    )

    assert len(dataset) == prev_size

    # prepend beginning-of-sentence token (<s>, equiv. to [CLS] in BERT)
    dataset = PrependTokenDataset(dataset, source_dictionary.bos())
    dataset = AppendTokenDataset(dataset, source_dictionary.eos())

    mask_whole_words = (
        get_whole_word_mask(args, source_dictionary)
        if mask_length != 'subword'
        else None
    )

    bpe = encoders.build_bpe(args)
    eoh = dictionary.indices[bpe.encode('</h>')]
    denoising_dataset = DenoisingDataset(
        dataset,
        dataset.sizes,
        dictionary,
        mask_idx,
        mask_whole_words,
        shuffle=False,
        seed=seed,
        args=args,
        eoh=eoh
    )

    for i in range(len(denoising_dataset)):
        ex = denoising_dataset[i]
