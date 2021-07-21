# Parser-Improved


Files:
	hw4_run.sh
		# Run all steps end-to-end
	hw4_topcfg.sh
	hw4_topcfg.py
	hw4_parser.sh
	hw4_parser.py
	hw4_improved_parser.sh

	Evaluation:
	hw4_timetest.sh

(Output Files)
	hw4_trained.pcfg
	parses_base.out
	parses_base.eval
	parses_improved.out
	parses_improved.eval

## 1. Induction

For HW4, We induced a Probabilistic Context-free Grammar (PCFG).
From a set of parsed sentences, it keeps a count of occurrences of each production rule (A) and a count of times A expands to BC.
To count the number of production rules, it stores the sentences as an nltk.Tree and then does a recursive depth-first search for Non-terminal rules.

It then calculates a probability for the rule (A -> BC):
 P(A -> BC) = Count(A -> BC)/ Count(A)

The output is a PCFG file

## 2. CKY to PCKY

For our HW4 parser (hw4_parser.sh), We altered the hw3_parser.py system.
First, the NoParsesException now passes and outputs a blank line.
We keep a table with the probabilities, which is attained for each sentence, when stored in a Tree, by multiplying the node probabilities.

For all the sentences that do have a parse, the system stores all the possible parses but loops through and finds the most probable parse.
The output is only the parse with the max probability (no probability in parse output).

## 3. Evaluation

$ $dir/evalb -p $dir/COLLINS.prm $dir/parses.gold parses_base.out > parses_base.eval

When compared to the gold parses, the evaluation of our system returns a tag accuracy of 99.06 and a F1 score of 88.27.

## 4. Improvements

We decided to look into improving the efficiency of the hw4_parser system without compromising the accuracy measures.
Since we only care about the best parse, our algorithm is unnecessarily storing and building all possible parses.

First, we considered Beam Thresholding, but the concern was that it does not always return the parse with the highest probability and may affect accuracy.
Since we were uncertain how much this might impact the accuracy, we altered our improvement to consider all unique LHS for every cell in the table but we only ever store the production with the highest cumulative probability for that LHS.

Improving table building:
When building at the table, when combining Left and Rights cells, we look at all possible productions from merging those cells.
We include with our ProbabilisticBackpointer the total probability of the parse up to that point.
When combining Left and Right cells, we multiply the production probability by the Left Backpointer's total probability and the Right Backpointer's total probability.
We continue this iteratively, building total probabilities until the algorithm finishes.
If there is a valid parse, then we will have stored the best total probability parse in the Start Node Backpointer.

We only keep the unique LHS productions with highest probability.

This improves table building and parse efficiency by reducing the amount of productions to consider when building the table.
Additionally, we have stored the best parse in the Start Node Backpointer. Whereas in our base system, we had stored multiple.

By making these improvements, we brought the average base table build time from ~0.007 seconds to the improved table build time of ~0.003
The average base parser time from ~0.008 to the improved parser time of ~0.0001.

This is done with no degradation in accuracy (See file: parses_improved.eval).
When we run evalb again on parses_improved.out, the improved system returns a tag accuracy of 99.06 and a F1 score of 88.27.


## 4.1 Efficiency Improvement Evaluations

With hw4_timetest.sh, one can compare the base table and parser run time with the improved versions.

To see the baseline time and the improved time (in seconds), run in the command line:
$ hw4_timetest.sh <input_PCFG_file> <test_sentence_filename>

$ bash hw4_timetest.sh hw4_trained.pcfg /data/sentences.txt

	Average base table build time:  0.007285621220415288
	Average base parser time:  0.008119505101984198


	Average improved table build time:  0.002638817646286704
	Average improved parser time:  0.00013316625898534602
