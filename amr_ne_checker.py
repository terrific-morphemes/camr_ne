"""Tools for analyzing Named Entities in Chinese AMR

Requires Python 3 and zhon to be installed

How to run:

1. Ensure that the variables in all caps point to the correct locations
2. Depending on which AMRs you want to compare,
    comment or uncomment the appropriate lines under if __name__ == "__main__"
3. Run from the command line (you probably want to print to a file):

python amr_ne_checker.py > results.txt
"""
#from __future__ import print_function
# -*- coding: utf-8 -*-
import os
import sys
import re
import csv

from zhon import hanzi  # for Chinese regex

sys.path.append('./camr/')
from amr import AMR
from smatch import get_amr_line
from preprocess import read_amrz

# Folder with the AMR data
DATA_DIR = os.path.join(os.curdir, 'data')

# Dictionary of Chinese NE tags and their English equivalents
CHINESE_ENTITIES = os.path.join(DATA_DIR, "chinese_entities.csv")

# All the gold AMRs
GOLD_AMRS = os.path.join(DATA_DIR, 'amr_zh_all.txt.amr')

# Gold AMRs from just the test set
GOLD_TEST = os.path.join(DATA_DIR, 'amr_zh_all.txt.test.amr')

# Parsed (i.e. by the model) AMRs with baseline features
BASIC_TEST = os.path.join(
    DATA_DIR,'amr_zh_all.txt.test.amr.basic_abt_feat.parsed')

# Parsed AMRs with sibling bigram features
SIBLING_BIGRAM_TEST = os.path.join(
    DATA_DIR,'amr_zh_all.txt.test.amr.sibling_bigram_feat.parsed')

# Parsed AMRs with sibling unigram features
SIBLING_TEST = os.path.join(
    DATA_DIR,'amr_zh_all.txt.test.amr.sibling_feat.parsed')

# Gold AMRs with preprocessing to normalize NE tags
NORMALIZED_NE_GOLD = os.path.join(
    DATA_DIR, "amr_zh_all_normalized_ne.txt.test.amr")

# Parsed AMRs with preprocessing to normalize NE tags
NORMALIZED_NE_PARSED = os.path.join(
    DATA_DIR, "amr_zh_all_normalized_ne.txt.test.amr.sibling_bigram_feat.parsed")


# Build the Chinese-to-English NE tag normalizer
zh_ne_dict = dict()
with open(CHINESE_ENTITIES) as source:
    reader = csv.DictReader(source)
    for row in reader:
        zh_ne_dict[row['Chinese']] = row['English']

def normalize_entity(entity):
    """Rename Chinese named entities to English"""
    if entity in ["coountry", "country"]: return "country"
    elif entity in ["peson"]: return "person"
    elif entity in zh_ne_dict.keys():
        return zh_ne_dict[entity]
    #elif len(re.findall("[{}]".format(hanzi.characters), entity)) > 0:
        #return "OtherChineseWord"
    else:
        return entity

def preprocess_nes(source_fname, dest_fname):
    """Preprocess AMRs by normalizing entity names
    Inputs:
        source_fname: source AMR file
        dest_fname: name of file to save to
    """
    rephrased_ids = set()
    rephrased_snt = None
    current_id = None
    with open(source_fname) as source:
        with open(dest_fname,'w') as dest:
            for line in source:
                if line.startswith("# ::id"):
                    current_id = line
                    dest.write(line)
                elif line.startswith("# ::snt"):
                    dest.write(line)
                elif line.startswith("# ::wid"):
                    dest.write(line)
                else:  # the amr graph
                    # Normalize the Chinese NE tags
                    line = re.sub(
                        r"[{}]+".format(hanzi.characters),
                        lambda x:normalize_entity(x.group()),
                        line
                    )
                    dest.write(line)

def count_named_entities(amrs):
    """Count each NE tag
    Inputs:
        amrs: list of AMRs
    Returns:
        dict with NE as keys and counts as values
    """
    entity_counts = dict()
    for amr in amrs:
        amr_graph = AMR.parse_AMR_line(amr)
        # variable to concept graph (from Damonte & Cohen)
        v2c = {}
        for n, v in zip(amr_graph.nodes, amr_graph.node_values):
            v2c[n] = v
        # relation, arg1, arg2 triples (from Damonte & Cohen)
        # e.g. (name, v1, v2) means "v1 is name of v2"
        # The indices are because triples() returns a list of lists
        triples = [t for t in amr_graph.get_triples()[1]]
        triples.extend([t for t in amr_graph.get_triples()[2]])
        named_entities = [str(v2c[v1]) for (l,v1,v2) in triples if l == "name"]
        for ne in named_entities:
            entity_counts[ne] = entity_counts.get(ne, 0) + 1
    return entity_counts

def evaluate_named_entities(gold_amr_file, parsed_amr_file, postprocessing=False):
    """Compare NE tagging for gold and parsed AMRs
    Inputs:
        gold_amr_file: file with the gold (human-annotated) AMRs
        parsed_amr_file: file with the parsed (machine-annotated) AMRs
        postprocessing: whether to normalize NE tags
    Returns:
        None (prints result)
    """
    print("Comparing named entities in gold {} vs parsed {}".format(
        gold_amr_file, parsed_amr_file
    ))
    if postprocessing is True:
        print("Performing postprocessing")
    else:
        print("Not performing postprocessing")

    gold_entity_counts = dict()
    parsed_entity_counts = dict()

    # Types of NE errors
    extra_ne_count = 0  # Parser has NE where gold has none
    missing_ne_count = 0  # Parser lacks NE where gold has one
    ne_mismatch_count = 0  # NE count matches but tags don't match
    perfect_match_count = 0  # NE count matches (could be zero)
    perfect_match_nonempty_count = 0  # NE count matches (nonzero) and tags too

    # Get gold amrs
    gold_comments_and_amrs = read_amrz(gold_amr_file)  # (comments, amrs)
    gold_comments = gold_comments_and_amrs[0]  # {'snt':snt,'id':id}
    gold_amrs = gold_comments_and_amrs[1]

    # Get parsed amrs
    parsed_comments_and_amrs = read_amrz(parsed_amr_file)  # (comments, amrs)
    parsed_comments = parsed_comments_and_amrs[0]  # {'snt':snt,'id':id}
    parsed_amrs = parsed_comments_and_amrs[1]

    # Keep track of all the entities
    all_gold_entities = list()
    all_parsed_entities = list()
    for i in range(len(gold_amrs)):
        gold_id = gold_comments[i]['id']
        gold_amr_graph = AMR.parse_AMR_line(gold_amrs[i])
        # variable to concept graph (from Damonte & Cohen)
        gold_v2c = {}
        for n, v in zip(gold_amr_graph.nodes, gold_amr_graph.node_values):
            gold_v2c[n] = v

        # relation, arg1, arg2 triples (from Damonte & Cohen)
        # e.g. (name, v1, v2) means "v1 is name of v2"
        # The indices are because triples() returns a list of lists
        gold_triples = [t for t in gold_amr_graph.get_triples()[1]]
        gold_triples.extend([t for t in gold_amr_graph.get_triples()[2]])
        gold_named_entities = [
            str(gold_v2c[v1]) for (l,v1,v2) in gold_triples if l == "name"
        ]

        # Normalize the NE tags if we're doing postprocessing
        if postprocessing is True:
            gold_named_entities = [
                normalize_entity(e) for e in gold_named_entities
            ]

        for ne in gold_named_entities:
            gold_entity_counts[ne] = gold_entity_counts.get(ne, 0) + 1

        # We're assuming the length of gold and parsed AMRs is the same
        # TODO: this is brittle and should be made more robust
        if len(parsed_amrs) >= i:
            # Get the parsed AMR corresponding to the gold AMR
            parsed_amr = parsed_amrs[i]
            parsed_amr_graph = AMR.parse_AMR_line(parsed_amr)
            # variable to concept graph (from Damonte & Cohen)
            parsed_v2c = {}
            for n, v in zip(
                        parsed_amr_graph.nodes, parsed_amr_graph.node_values):
                parsed_v2c[n] = v
            # relation, arg1, arg2 triples (from Damonte & Cohen)
            # e.g. (name, v1, v2) means "v1 is name of v2"
            # The indices are because triples() returns a list of lists
            parsed_triples = [t for t in parsed_amr_graph.get_triples()[1]]
            parsed_triples.extend(
                            [t for t in parsed_amr_graph.get_triples()[2]])
            parsed_named_entities = [
                        str(parsed_v2c[v1]) for (l,v1,v2) in parsed_triples
                        if l == "name"
            ]

            # Normalize the NE tags if we're doing postprocessing
            if postprocessing is True:
                parsed_named_entities = [
                    normalize_entity(e) for e in parsed_named_entities
            ]

            for ne in parsed_named_entities:
                parsed_entity_counts[ne] = parsed_entity_counts.get(ne, 0) + 1

            # Get the various error counts
            if len(gold_named_entities) < len(parsed_named_entities):
                extra_ne_count += 1
            elif len(gold_named_entities) > len(parsed_named_entities):
                missing_ne_count += 1
            elif gold_named_entities == parsed_named_entities and \
                    len(gold_named_entities) > 0:
                perfect_match_nonempty_count += 1
            elif gold_named_entities == parsed_named_entities:
                perfect_match_count += 1
            else:
                ne_mismatch_count += 1

            # If the lists of entities are different, add "None"
            while len(gold_named_entities) < len(parsed_named_entities):
                gold_named_entities.append("None")
            while len(parsed_named_entities) < len(gold_named_entities):
                parsed_named_entities.append("None")

            # Add to the total lists of NEs
            all_gold_entities.extend(gold_named_entities)
            all_parsed_entities.extend(parsed_named_entities)

    print("Extra NEs: {}".format(extra_ne_count))
    print("Missing NEs: {}".format(missing_ne_count))
    print("Mismatch NEs: {}".format(ne_mismatch_count))
    print("Perfect (nonempty) match: {}".format(perfect_match_nonempty_count))
    print()

if __name__ == "__main__":
    evaluate_named_entities(GOLD_TEST, BASIC_TEST)
    evaluate_named_entities(GOLD_TEST, BASIC_TEST, postprocessing=True)

    evaluate_named_entities(GOLD_TEST, SIBLING_TEST)
    evaluate_named_entities(GOLD_TEST, SIBLING_TEST, postprocessing=True)

    evaluate_named_entities(GOLD_TEST, SIBLING_BIGRAM_TEST)
    evaluate_named_entities(GOLD_TEST, SIBLING_BIGRAM_TEST, postprocessing=True)

    evaluate_named_entities(NORMALIZED_NE_GOLD, NORMALIZED_NE_PARSED)
