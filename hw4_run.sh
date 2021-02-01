#!/bin/bash

# Call all components of the system:
# hw4_run.sh  $1 <treebank_filename>  $2 <output_PCFG_file> \
# $3 <test_sentence_filename> $4 <baseline_parse_output_filename> \
# $5 <input_PCFG_file> \
# $6 <improved_parse_output_filename> \
# $7 <baseline_eval>  $8 <improved_eval>

bash hw4_topcfg.sh $1 > $2

bash hw4_parser.sh $2 $3 > $4

bash hw4_improved_parser.sh $2 $3 > $6

#Evals
/dropbox/20-21/571/hw4/tools/evalb -p /dropbox/20-21/571/hw4/tools/COLLINS.prm /dropbox/20-21/571/hw4/data/parses.gold $4 > $7
/dropbox/20-21/571/hw4/tools/evalb -p /dropbox/20-21/571/hw4/tools/COLLINS.prm /dropbox/20-21/571/hw4/data/parses.gold $6 > $8


# Run to obtain files:
# $ bash hw4_run.sh /dropbox/20-21/571/hw4/data/parses.train hw4_trained.pcfg /dropbox/20-21/571/hw4/data/sentences.txt parses_base.out input_PCFG_file parses_improved.out parses_base.eval parses_improved.eval
