import itertools
import json
import os
import re
import sys

import pandas as pd


GPT_SPACE_CHAR = '\u0120'


def create_note_tok(note_str, prefix=True):
    note_str_clean = re.sub('[\W+]+', '_', note_str.strip()).lower()
    if prefix:
        return 'note_type_' + note_str_clean
    return note_str_clean


def create_concept_tok(concept_str):
    return str(len(concept_str)) + '_concept_' + concept_str.lower()


def add_special_toks(encoder_fn, dict_fn, unigram_fn, vocab_fn, add_concepts=False):
    unigram_counts_df = pd.read_csv(unigram_fn)
    unigram_counts_df.dropna(inplace=True)
    DO_NOT_REMOVE_NO_SPACE = unigram_counts_df['tok'].tolist()
    DO_NOT_REMOVE_W_SPACE = [GPT_SPACE_CHAR + x for x in DO_NOT_REMOVE_NO_SPACE]

    with open(vocab_fn, "r", encoding="utf-8") as f:
        bpe_data = f.read()
    bpe_merges = list(itertools.chain(*[merge_str.split() for merge_str in bpe_data.split("\n")[1:-1]]))
    DO_NOT_REMOVE = set(DO_NOT_REMOVE_NO_SPACE + DO_NOT_REMOVE_W_SPACE + bpe_merges)
    DO_NOT_REMOVE.add('<-')
    DO_NOT_REMOVE.add(GPT_SPACE_CHAR + '<-')
    SPECIAL_TOKS = []

    PHI_TOKS = [
        'phi_sn',
        'phi_id',
        'phi_mrn',
        'phi_ssn',
        'phi_age',
        'phi_phone',
        'phi_pager',
        'phi_contact',
        'phi_provider_phone',
        'phi_name',
        'phi_dictator',
        'phi_attending',
        'phi_hospital',
        'phi_loc',
        'phi_address',
        'phi_country',
        'phi_state',
        'phi_uni',
        'phi_year',
        'phi_month',
        'phi_dt',
        'phi_month_year',
        'phi_month_day_year',
        'phi_clip_num',
        'phi_holiday',
        'phi_company',
        'phi_job_num',
        'phi_unit_num',
        'phi_url',
        'phi_other'
    ]

    SPECIAL_TOKS += PHI_TOKS
    for phi_tok in PHI_TOKS:
        SPECIAL_TOKS.append(GPT_SPACE_CHAR + phi_tok)

    MIMIC_NOTE_TYPES = [
        'nursing_other',
        'radiology',
        'nursing',
        'ecg',
        'physician',
        'discharge_summary',
        'echo',
        'respiratory',
        'nutrition',
        'general',
        'rehab_services',
        'social_work',
        'case_management',
        'pharmacy',
        'consult'
    ]

    nts = [create_note_tok(x) for x in MIMIC_NOTE_TYPES]
    SPECIAL_TOKS += nts

    TAGS = ['d', 'h']  # , 'p', 'e']
    start_template = '<{}>'
    end_template = '</{}>'

    for tag in TAGS:
        s, e = start_template.format(tag), end_template.format(tag)
        SPECIAL_TOKS.append(s)
        SPECIAL_TOKS.append(e)

        # SPECIAL_TOKS.append(GPT_SPACE_CHAR + s)
        # SPECIAL_TOKS.append(GPT_SPACE_CHAR + e)

    if add_concepts:
        concepts_fn = os.path.expanduser('~/pores/shared_preprocess/sec_tag/data/concepts_and_synonyms.txt')
        cols = ['cid', 'concept_name', 'synonym_id', 'synonym_name', 'synonym_type']
        concepts_df = pd.read_csv(concepts_fn, sep='\t', names=cols)
        concepts_df.dropna(subset=['concept_name'], inplace=True)
        SECTION_CONCEPTS = concepts_df['concept_name'].apply(create_concept_tok).unique().tolist()
        SPECIAL_TOKS += SECTION_CONCEPTS

    with open(encoder_fn, 'r') as fd:
        encoder = json.load(fd)
    orig_n = len(encoder)
    decoder = {v: k for k, v in encoder.items()}
    tok_counts = []
    with open(dict_fn, 'r') as fd:
        count_lines = fd.readlines()
        for line in count_lines:
            line = line.strip()
            if len(line) == 0:
                continue
            id, count = line.split(' ')
            try:
                tok = decoder[int(id)]
                if 'endoftext' in tok or tok in DO_NOT_REMOVE:
                    continue
                tok_counts.append({'tok': tok, 'count': int(count)})
            except:
                print('Can\'t replace {}'.format(id))

    tok_counts_df = pd.DataFrame(tok_counts)
    tok_counts_df.sort_values(by='count', ascending=True, inplace=True)

    print('{} tokens do not appear in mimic.'.format(len(tok_counts_df)))
    toks_to_add = list(set(SPECIAL_TOKS))
    toks_to_add = [x for x in toks_to_add if x not in encoder]
    add_n = len(toks_to_add)
    print('Replacing {} unused tokens with special'.format(add_n))

    toks_to_remove_df = tok_counts_df[:add_n]
    toks_to_remove = toks_to_remove_df['tok'].tolist()
    for new_key, old_key in zip(toks_to_add, toks_to_remove):
        assert new_key not in encoder
        encoder[new_key] = encoder.pop(old_key)
    assert orig_n == len(encoder)
    return encoder, toks_to_remove_df, toks_to_add


if __name__ == '__main__':
    tok_dir = os.path.expanduser('~/fairseq/data/gpt2')
    encoder_fn = os.path.join(tok_dir, 'encoder.json')
    dict_fn = os.path.join(tok_dir, 'dict.txt')
    vocab_fn = os.path.join(tok_dir, 'vocab.bpe')
    unigram_fn = os.path.expanduser('~/pores/mimic_preprocess/stats/unigram_counts.csv')

    new_encoder, replaced_df, added_toks = add_special_toks(encoder_fn, dict_fn, unigram_fn, vocab_fn)
    new_encoder_fn = os.path.join(tok_dir, 'encoder_special_toks.json')
    print('Dumping {} toks to {}'.format(len(new_encoder), new_encoder_fn))
    with open(new_encoder_fn, 'w') as fd:
        json.dump(new_encoder, fd)

    replaced_fn = os.path.join(tok_dir, 'replaced_toks.csv')
    rn = len(replaced_df)
    print('Dumping {} replaced tokens to {}. Should be removed before BPE encoding...'.format(rn, replaced_fn))
    replaced_df.to_csv(replaced_fn, index=False)

    added_fn = os.path.join(tok_dir, 'added_toks.csv')
    added_df = pd.DataFrame({'tok': added_toks})
    print('Dumping {} added tokens to {}'.format(len(added_df), added_fn))
    added_df.to_csv(added_fn, index=False)
