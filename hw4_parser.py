#!/usr/bin/env python3
import sys
import nltk
from collections import namedtuple, defaultdict
from nltk.tree import Tree, ProbabilisticTree
from nltk.grammar import PCFG, ProbabilisticProduction
import argparse
import heapq
import timeit
import time
# import copy


Backpointer = namedtuple('Backpointer', ['prod', 'l_child', 'r_child'])
ProbabilisticBackpointer = namedtuple('Backpointer', ['prob','prod', 'l_child', 'r_child'])


def get_all_possible_pairs(grammar, left_cell, right_cell):
    both = set()
    productions = set()
    #print(left_cell)
    #print(right_cell)

    for lc in left_cell.keys():
        productions.update(set(grammar.productions(rhs=lc)))

    both = {p for p in productions if p.rhs()[1] in right_cell}

    return both

def get_all_possible_pairs_improved(grammar, table, i, k, j):
    """Improving efficiency of table building
    """
    left_cell = table[i][k]
    right_cell = table[k][j]
    new_cell = table[i][j]

    for rhs_l in left_cell.keys():
        for prod in grammar.productions(rhs=rhs_l):
            lhs = prod.lhs()
            rhs_r = prod.rhs()[1]
            if rhs_r not in right_cell:
                continue
            probability = prod.prob() * left_cell[rhs_l].prob * right_cell[rhs_r].prob
            if lhs in new_cell:
                if new_cell[lhs].prob < probability:
                    new_cell[lhs] = ProbabilisticBackpointer(probability, prod, (i, k), (k, j))
            else:
                new_cell[lhs] = ProbabilisticBackpointer(probability, prod, (i, k), (k, j))


def cky_build_table(tokens, grammar):
    n = len(tokens)
    table = [[defaultdict(set) for j in range(n+1)] for i in range(n+1)]  # matrix = (n+1) * (n+1)

    for j in range(1, len(tokens) + 1):
        word = tokens[j-1]  # sentence indices start at 0, so offset by 1
        for prod in grammar.productions(rhs=word):  # All rules that point to terminals → words[j] ∈ grammar }
            table[j-1][j][prod.lhs()].add(Backpointer(prod, None, None))
        for i in range(j - 2, -1, -1):
            for k in range((i + 1), (j)):
                prods = get_all_possible_pairs(grammar, table[i][k], table[k][j])
                for p in prods:
                    table[i][j][p.lhs()].add(Backpointer(p, (i, k), (k, j)))
                # table[i,j]←table[i,j] ∪{A|A → BC ∈ grammar, B ∈ table[i,k],C ∈ table[k, j] }
                # NT rules corresponding with table[i,k] table[k, j]
    return table


def cky_build_table_improved(tokens, grammar):
    n = len(tokens)
    table = [[dict() for j in range(n+1)] for i in range(n+1)]  # matrix = (n+1) * (n+1)

    for j in range(1, len(tokens) + 1):
        word = tokens[j-1]  # sentence indices start at 0, so offset by 1
        for prod in grammar.productions(rhs=word):  # All rules that point to terminals → words[j] ∈ grammar }
            lhs = prod.lhs()
            if lhs not in table[j-1][j]:
                table[j-1][j][lhs] = ProbabilisticBackpointer(prod.prob(), prod, None, None)
            else:
                if table[j-1][j][lhs].prob < prod.prob():
                    table[j-1][j][lhs] = ProbabilisticBackpointer(prod.prob(), prod, None, None)
        for i in range(j - 2, -1, -1):
            for k in range((i + 1), (j)):
                get_all_possible_pairs_improved(grammar, table, i, k, j)

                # table[i,j]←table[i,j] ∪{A|A → BC ∈ grammar, B ∈ table[i,k],C ∈ table[k, j] }
                # NT rules corresponding with table[i,k] table[k, j]
    return table
# Every cell[i,j] Also, Add to cell Parent, if there is one.


class NoParsesException(ValueError):
    pass


def parse_table_orig(grammar, table):
    starts = table[0][len(table)-1][grammar.start()]
    if not starts:
        raise NoParsesException("No possible parses")

    def get_subparse(backpointer):
        parses = []
        probability = backpointer.prod.prob()  # Probability of tree for input S, P(T, S) = P(T)P(S|T) = P(T)
        if backpointer.l_child is None and backpointer.r_child is None:
            return [ProbabilisticTree(str(backpointer.prod.lhs()),
                    [str(backpointer.prod.rhs()[0])],
                    prob=probability)
                    ]

        left, left_index = backpointer.prod.rhs()[0], backpointer.l_child
        left_parses = []
        for b in table[left_index[0]][left_index[1]][left]:
            left_parses += get_subparse(b)

        right, right_index = backpointer.prod.rhs()[1], backpointer.r_child
        right_parses = []
        for b in table[right_index[0]][right_index[1]][right]:
            right_parses += get_subparse(b)

        for left_parse in left_parses:
            for right_parse in right_parses:

                parses.append(ProbabilisticTree(str(backpointer.prod.lhs()),
                                                [left_parse, right_parse],
                                                prob=probability*left_parse.prob()*right_parse.prob()))
        return parses

    parses = []
    for s in starts:
        parses += get_subparse(s)
    return parses


def parse_table_improved(grammar, table):

    if grammar.start() not in table[0][len(table)-1]:
        raise NoParsesException("No possible parses")

    def get_subparse(backpointer):
        probability = backpointer.prob  # Probability of tree for input S, P(T, S) = P(T)P(S|T) = P(T)
        if backpointer.l_child is None and backpointer.r_child is None:
            return ProbabilisticTree(str(backpointer.prod.lhs()),
                    [str(backpointer.prod.rhs()[0])],
                    prob=probability)


        left, left_index = backpointer.prod.rhs()[0], backpointer.l_child
        right, right_index = backpointer.prod.rhs()[1], backpointer.r_child

        left_cell = table[left_index[0]][left_index[1]]
        right_cell = table[right_index[0]][right_index[1]]

        left_parse = get_subparse(left_cell[left])
        right_parse = get_subparse(right_cell[right])

        return ProbabilisticTree(str(backpointer.prod.lhs()),
                                        [left_parse, right_parse],
                                        prob=probability)

    start = table[0][len(table)-1][grammar.start()]
    return get_subparse(start)


def print_table(table):
    for i in range(len(table)):
        print('Table row ', i)
        for j in range(len(table) - i):
            print('\t\t', j, table[i][j])


def parse_args(args=None):
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(prog='hw4_parser', add_help=True) # prog  + --help
    parser.add_argument('input_PCFG_file', action='store', help="PCFG file")
    parser.add_argument('test_sentence_filename', action='store', type=str,
                        help="Input file of sentences to parse")
    parser.add_argument('-i', '--improved', action='store_true', help="Runs improved version of parser")
    parser.add_argument('-b', '--beam-width', action='store', type=int, choices=range(1, 1001),
                        default=5, metavar="[1-1000]", help="Number of choices to consider for Beam Search Algorithm")
    parser.add_argument('-t', '--timeit', action='store_true', help="Timed improved vs. base")
    return parser.parse_args(args)


def main(args):

    grammar = nltk.data.load(args.input_PCFG_file)
    # print(grammar._rhs_index)
    tot_sentences = 0
    tot_parses = 0
    with open(args.test_sentence_filename) as f:
        for line in f:
            line = line.strip()
            # print(line)
            token_list = nltk.word_tokenize(line)
            if args.improved:
                table = cky_build_table_improved(token_list, grammar)
            else:
                table = cky_build_table(token_list, grammar)
            # print_table(table)
            try:
                if args.improved:
                    parses = [parse_table_improved(grammar, table)]
                else:
                    parses = parse_table_orig(grammar, table)
                max_parse = None
                max_prob = -1
                for p in parses:
                    if p.prob() > max_prob:
                        max_prob = p.prob()
                        max_parse = p
                # print(str(max_parse).replace("\n", " ")) # for best parse and probability
                print(Tree.__str__(max_parse).replace("\n", " ")) # for best parse only
                # print(f"Number of possible parses: {len(parses)}", "\n")
                tot_parses += len(parses)
            except NoParsesException:
                print("") # Print blank
            tot_sentences += 1

        # print("Average number of parses per sentence: ", tot_parses / tot_sentences)

def timed_run(grammar, args):
    tot_sentences = 0
    tot_parses = 0
    tot_table_time = time.perf_counter() - time.perf_counter()
    tot_parse_time = time.perf_counter() - time.perf_counter()
    with open(args.test_sentence_filename) as f:
        for line in f:
            line = line.strip()
            # print(line)
            token_list = nltk.word_tokenize(line)
            start_time = time.perf_counter()
            if args.improved:
                table = cky_build_table_improved(token_list, grammar)
            else:
                table = cky_build_table(token_list, grammar)
            tot_table_time += time.perf_counter() - start_time
            # print_table(table)
            try:
                start_time = time.perf_counter()
                if args.improved:
                    parses = [parse_table_improved(grammar, table)]
                else:
                    parses = parse_table_orig(grammar, table)
                tot_parse_time += time.perf_counter() - start_time
                max_parse = None
                max_prob = -1
                for p in parses:
                    #print(p)
                    if p.prob() > max_prob:
                        max_prob = p.prob()
                        max_parse = p
                # print(str(max_parse).replace("\n", " ")) # for best parse and probability
                # print(Tree.__str__(max_parse).replace("\n", " ")) # for best parse only
                # print(f"Number of possible parses: {len(parses)}", "\n")
                tot_parses += len(parses)
            except NoParsesException:
                print("") # Print blank
            tot_sentences += 1

        return tot_table_time/tot_sentences, tot_parse_time/tot_sentences

def timetest(args):
    """For testing purposes
    """
    tot_table_time = 0
    tot_parse_time = 0
    grammar = nltk.data.load(args.input_PCFG_file)
    args.improved = False
    table_time, parse_time = timed_run(grammar, args)
    print('Average base table build time: ', table_time)
    print('Average base parser time: ', parse_time)

    args.improved = True
    table_time, parse_time = timed_run(grammar, args)
    print('Average improved table build time: ', table_time)
    print('Average improved parser time: ', parse_time)
    # print(grammar._rhs_index)


if __name__ == '__main__':
    args = parse_args()
    if args.timeit:
        timetest(args)
    else:
        main(args)
    #test()
