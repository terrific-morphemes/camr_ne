"""Tools for analyzing Named Entities in Chinese AMR"""
#from __future__ import print_function
# -*- coding: utf-8 -*-
import os
import re
import random

import seaborn as sns
from zhon import hanzi
from pandas_ml import ConfusionMatrix
import matplotlib.pyplot as plt


from amr import AMR
from smatch import get_amr_line


from preprocess import read_amrz


DATA_DIR = os.path.join(os.curdir, 'data')
RESULTS_DIR = os.path.join(os.curdir,'results')
#path to training AMRs
GOLD_AMRS = os.path.join(DATA_DIR, 'amr_zh_all.txt.amr')
GOLD_TEST = os.path.join(DATA_DIR, 'amr_zh_all.txt.test.amr')
REPHRASED_GOLD = os.path.join(DATA_DIR, "amr_zh_all_rephrased.txt.amr")
GOLD_SMALL = os.path.join(DATA_DIR, 'amr_small.txt')

PARSED_AMR_SMALL = os.path.join(DATA_DIR, 'rephrased_small.txt')
BASIC_TEST = os.path.join(DATA_DIR,'amr_zh_all.txt.test.amr.basic_abt_feat.parsed')
REPHRASED_TEST = os.path.join(DATA_DIR,'amr_zh_all_rephrased.txt.test.amr.basic_abt_feat.parsed')
SIBLING_TEST = os.path.join(DATA_DIR,'amr_zh_all.txt.test.amr.sibling_feat.parsed')
SIBLING_BIGRAM_TEST = os.path.join(DATA_DIR,'amr_zh_all.txt.test.amr.sibling_bigram_feat.parsed')

"""
# CAMR (Chinese Abstract Meaning Representation) release v0.1
# generated on 2017-06-16 22:58:20
# ::id export_amr.1 ::2017-02-10 13:29:53
# ::snt 最近 ， 我们 通过 知情 人士 从 衡阳市 殡葬 管理处 财务 部门 复印 出 部分 原始 发票 凭证 11 份 （ 共计 有 30余 份 ） 。
# ::wid x1_最近 x2_， x3_我们 x4_通过 x5_知情 x6_人士 x7_从 x8_衡阳市 x9_殡葬 x10_管理处 x11_财务 x12_部门 x13_复印 x14_出 x15_部分 x16_原始 x17_发票 x18_凭证 x19_11 x20_份 x21_（ x22_共计 x23_有 x24_30余 x25_份 x26_） x27_。 x28_
"""

NE = ["city-district", "country", "country-region", "ethnic-group", "government-organization", "location", "newspaper", "organization", "person"]
def normalize_entity(entity):
    """Rename Chinese named entities to English"""
    if entity in ["coountry", "country"]: return "country"
    if entity in ["peson"]: return "person"
    if len(re.findall("[{}]".format(hanzi.characters), entity)) > 0:
        return "ChineseWord"
    else:
        return entity

def count_named_entities(amrs):
    """Get all the named entities
    Inputs:
        amrs: list of AMRs
    Returns:
        dict with NE as keys and counts as values
    """
    entity_counts = dict()
    for amr in amrs:
        amr_graph = AMR.parse_AMR_line(amr)
        # amr_evaluation var2concept
        v2c = {}
        for n, v in zip(amr_graph.nodes, amr_graph.node_values):
            v2c[n] = v
        # print(v2c)
        # I don't know why we need these indices but we do
        triples = [t for t in amr_graph.get_triples()[1]]
        triples.extend([t for t in amr_graph.get_triples()[2]])
        #print(triples)
        # named_ent(v2c, triples)
        # "v1 is name of v2"
        named_entities = [str(v2c[v1]) for (l,v1,v2) in triples if l == "name"]
        for ne in named_entities:
            entity_counts[ne] = entity_counts.get(ne, 0) + 1
    return entity_counts

def evaluate_named_entities(gold_amr_file, parsed_amr_file):
    """Compare NE tagging for gold and parsed AMRs
    Inputs:
        gold_amr_file: file with the gold (human-annotated) AMRs
        parsed_amr_file: file with the parsed (machine-annotated) AMRs
    Returns:
        list of (id, snt, amr) tuples
    """
    gold_entity_counts = dict()
    parsed_entity_counts = dict()

    extra_ne_count = 0
    missing_ne_count = 0
    ne_mismatch_count = 0
    perfect_match_count = 0
    perfect_match_nonempty_count = 0

    chinese_entities = list()
    match_amrs = list() #(id,snt)
    gold_comments_and_amrs = read_amrz(gold_amr_file) #(comment_list, amr_list)
    gold_comments = gold_comments_and_amrs[0] #{'snt','id'}
    gold_amrs = gold_comments_and_amrs[1]

    parsed_comments_and_amrs = read_amrz(parsed_amr_file) #(comment_list, amr_list)
    parsed_comments = parsed_comments_and_amrs[0] #{'snt','id'}
    parsed_amrs = parsed_comments_and_amrs[1]

    all_gold_entities = list()
    all_parsed_entities = list()
    for i in range(len(gold_amrs)):
        gold_id = gold_comments[i]['id']
        gold_amr_graph = AMR.parse_AMR_line(gold_amrs[i])
        # amr_evaluation var2concept
        gold_v2c = {}
        for n, v in zip(gold_amr_graph.nodes, gold_amr_graph.node_values):
            gold_v2c[n] = v
        # print(v2c)
        # I don't know why we need these indices but we do
        gold_triples = [t for t in gold_amr_graph.get_triples()[1]]
        gold_triples.extend([t for t in gold_amr_graph.get_triples()[2]])
        #print(triples)
        # named_ent(v2c, triples)
        # "v1 is name of v2"
        gold_named_entities = [str(gold_v2c[v1]) for (l,v1,v2) in gold_triples if l == "name"]
        for ne in gold_named_entities:
            gold_entity_counts[ne] = gold_entity_counts.get(ne, 0) + 1

        gold_renamed_entities = [normalize_entity(e) for e in gold_named_entities]
        #gold_chinese_entities = [e for e in gold_named_entities if len(re.findall("[{}]".format(hanzi.characters), e)) > 0]
        #chinese_entities.extend(gold_chinese_entities)

        """
        parsed_graph_index = -1
        for index, comment in enumerate(parsed_comments):
            if comment['id'] == gold_id:
                parsed_graph_index = index
        if parsed_graph_index != -1:
        """
        if len(parsed_amrs) >= i:
            parsed_amr = parsed_amrs[i]
            #parsed_amr = parsed_amrs[parsed_graph_index]
            parsed_amr_graph = AMR.parse_AMR_line(parsed_amr)
            parsed_v2c = {}
            for n, v in zip(parsed_amr_graph.nodes, parsed_amr_graph.node_values):
                parsed_v2c[n] = v
            # I don't know why we need these indices but we do
            parsed_triples = [t for t in parsed_amr_graph.get_triples()[1]]
            parsed_triples.extend([t for t in parsed_amr_graph.get_triples()[2]])
            # named_ent(v2c, triples)
            parsed_named_entities = [str(parsed_v2c[v1]) for (l,v1,v2) in parsed_triples if l == "name"]
            for ne in parsed_named_entities:
                parsed_entity_counts[ne] = parsed_entity_counts.get(ne, 0) + 1

            #parsed_chinese_entities = [e for e in parsed_named_entities if len(re.findall("[{}]".format(hanzi.characters), e)) > 0]
            #chinese_entities.extend(parsed_chinese_entities)
            parsed_renamed_entities = [normalize_entity(e) for e in parsed_named_entities]
            if len(gold_renamed_entities) < len(parsed_renamed_entities):
                extra_ne_count += 1
            elif len(gold_renamed_entities) > len(parsed_renamed_entities):
                missing_ne_count += 1
            elif gold_renamed_entities == parsed_renamed_entities:
                perfect_match_count += 1
            elif gold_renamed_entities == parsed_renamed_entities and len(gold_renamed_entities) > 0:
                perfect_match_nonempty_count += 1
            else:
                ne_mismatch_count += 1
            while len(gold_renamed_entities) < len(parsed_renamed_entities):
                gold_renamed_entities.append("None")
            while len(parsed_renamed_entities) < len(gold_renamed_entities):
                parsed_renamed_entities.append("None")

            while len(gold_named_entities) < len(parsed_named_entities):
                gold_named_entities.append("None")
            while len(parsed_named_entities) < len(gold_named_entities):
                parsed_named_entities.append("None")


            #print("Gold: {} Parsed {}".format(gold_renamed_entities, parsed_renamed_entities))
            #print("Gold: {} Parsed {}".format(gold_named_entities, parsed_named_entities))

            all_gold_entities.extend(gold_renamed_entities)
            all_parsed_entities.extend(parsed_renamed_entities)

            #all_gold_entities.extend(gold_named_entities)
            #all_parsed_entities.extend(parsed_named_entities)

            """
            gold_named_nodes = [str(gold_v2c[v2]) for (l, v1, v2) in gold_triples if l == "name"]
            for gold_node in gold_named_nodes:
                print("Looking for {} in parsed".format(gold_node))
                parsed_nodes = [v2 for (l, v1, v2) in parsed_triples if v2 == gold_node]
                if len(parsed_nodes) > 0:
                    parsed_node = parsed_nodes[0]
                    node_comparisons.append((gold_node, gold_v2c[gold_node], parsed_v2c[gold_node]))
            """
    print("Extra NEs: {} Missing NEs: {} Mismatch: {} Perfect match: {} Perfect match nonempty: {}".format(
            extra_ne_count, missing_ne_count, ne_mismatch_count, perfect_match_count, perfect_match_nonempty_count
    ))
    gold_only_nes = list()
    parsed_only_nes = list()
    all_ne_keys = set(gold_entity_counts.keys()).union(set(parsed_entity_counts.keys()))
    for ne in sorted(list(all_ne_keys)):
        gold_ne_count = gold_entity_counts.get(ne, 0)
        parsed_ne_count = parsed_entity_counts.get(ne, 0)
        if gold_ne_count != 0 and parsed_ne_count == 0: gold_only_nes.append(ne)
        if gold_ne_count == 0 and parsed_ne_count != 0: parsed_only_nes.append(ne)

        print("{}: Gold {} Parsed {}".format(ne, gold_ne_count, parsed_ne_count))
    print("Entities only present in gold: {}".format(gold_only_nes))
    print("Entities only present in parsed: {}".format(parsed_only_nes))

    #confusion_matrix = ConfusionMatrix(all_gold_entities, all_parsed_entities)
    #confusion_matrix.plot(backend='seaborn')
    #sns.set_palette("husl")
    #plt.show()

        #print(amr_graph.node_values[0])
        #node_values = amr_graph.node_values
        #if concept in node_values:
            #match_amrs.append((comments[i]['id'],comments[i]['snt'],amrs[i]))
            #possible_ids.append((comments[i]['id'].encode('utf8'),comments[i]['snt'].encode('utf8'),amrs[i].encode('utf8')))
    #print("Total number of AMRs with '{}': {}".format(concept,len(match_amrs)))
    #return sorted(match_amrs,key=lambda x: int(x[0].split(' ')[0].split('.')[1])) #sort by id number

if __name__ == "__main__":
    #evaluate_named_entities(GOLD_TEST, BASIC_TEST)
    #evaluate_named_entities(GOLD_TEST, SIBLING_TEST)
    evaluate_named_entities(GOLD_TEST, SIBLING_BIGRAM_TEST)
