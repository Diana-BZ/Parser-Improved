#!/usr/bin/env python3
import sys
import nltk
from collections import namedtuple, defaultdict
from nltk.tree import Tree
from nltk.grammar import is_terminal, CFG, Production, Nonterminal


def get_child_names(node):
    names = []
    for child in node:
        if isinstance(child, Tree):
            names.append(Nonterminal(child._label))
        else:
            names.append(child)
    return names

def update_dictionary(lhs_dict, whole_dict, node):
    production = Production(Nonterminal(node.label()), get_child_names(node))
    if production.lhs() not in lhs_dict:
        lhs_dict[production.lhs()] = 0
    if production not in whole_dict:
        whole_dict[production] = 0

    lhs_dict[production.lhs()] += 1
    whole_dict[production] += 1


def travel_tree(lhs_dict, whole_dict, tree):
    """Traverses tree with recursive depth first search and updates rule counts for each A, and A -> BC"""

    def travel_subtree(lhs_dict, whole_dict, node):
        if not isinstance(node, Tree):
            return
        update_dictionary(lhs_dict, whole_dict, node)
        for child in node:
            travel_subtree(lhs_dict, whole_dict, child)

    root = tree  # tree is just the root node of the entire tree structure
    travel_subtree(lhs_dict, whole_dict, root)


def get_rule_counts(treebank_file):
    """Gets the count of Rules in RHS and LHS for a tree/line"""
    lhs_dict = {}
    whole_dict = {}
    with open(treebank_file) as f:
        for line in f:
            line = line.strip()
            tree = Tree.fromstring(line)
            travel_tree(lhs_dict, whole_dict, tree)
    return lhs_dict, whole_dict


def calculate_probability(lhs_dict, whole_dict, production_rule):
    """Takes a rule and gives back a probability."""
    lhs = production_rule.lhs()
    probability = whole_dict[production_rule] / lhs_dict[lhs]
    return probability


def main(treebank_file):
    lhs_dict, whole_dict = get_rule_counts(treebank_file)
    for production_rule in whole_dict.keys():
        probability = calculate_probability(lhs_dict, whole_dict, production_rule)
        print(production_rule, f"[{probability}]")


if __name__ == "__main__":
    treebank_file = sys.argv[1]
    # output_PCFG = sys.argv[2]
    main(treebank_file)
