# Information extraction final project
Nicholas A Miller
nam37@brandeis.edu
24 Apr 2018

## Purpose

Tools for evaluating named entity recognition in Chinese Abstract Meaning Representation using the CAMR parser.

## Scripts

`amr_ne_checker.py`: main tool for NE evaluation

## Data

### In the `data` folder:
- `amr_ne_en.txt`: list of English AMR named entities
- `chinese_entities.csv`: For normalizing Chinese NE tags
- `amr_zh_all.txt`: All the gold AMRs
- `amr_zh_all.txt.amr`: All the gold AMRs (alternate format)
- `amr_zh_all.txt.test.amr`: All the gold AMRs (test set only)
- `amr_zh_all.txt.test.amr.basic_abt_feat.parsed`: Parsed AMRs (test set), baseline features
- `amr_zh_all.txt.test.amr.sibling_feat.parsed`: Parsed AMRs (test set), baseline + sibling unigram features
- `amr_zh_all.txt.test.amr.sibling_bigram_feat.parsed`: Parsed AMRs (test set), baseline + sibling unigram + bigram features
- `amr_zh_all_normalized_ne.txt`: Gold AMRs with preprocessing to normalize NEs (all 10k AMRs)
- `amr_zh_all_normalized_ne.txt.test.amr`: Gold AMRs with preprocessing to normalize NEs (just the test set)
- `amr_zh_all_normalized_ne.txt.test.amr.sibling_bigram_feat.parsed`: parsed AMRs with preprocessing to normalize NEs (just the test set)

### In the `camr` folder:

The following files are pieces of Chuan Wang's CAMR parser (https://github.com/c-amr/camr):

- `Aligner.py`
- `amr_graph.py`
- `constants.py`
- `data.py`
- `preprocess.py`
- `span_graph.py`
- `span.py`
- `util.py`

The following file is from Damonte & Cohen's AMR evaluation toolkit (https://github.com/mdtux89/amr-evaluation):

- `smatch.py`

Just enough is included to run `amr_ne_checker.py`. I haven't altered any of their code except to make sure that the import statements work properly. For details of how their code works, consult their repositories.

## Notes about what's not included

This folder does not contain the entire CAMR parser. To retrain the model yourself, download the parser and train it on the whole dataset according to the instructions here:

https://github.com/c-amr/camr

This folder also does not contain the entire AMR-evaluation toolkit. To perform full evaluation, e.g. to get Smatch scores, compare gold and parsed test sets according to the instructions here:

https://github.com/mdtux89/amr-evaluation

## How to run

Run `amr_ne_checker.py` according to the instructions at the top of that file.
