# Dependency Parser

This repository contains code to train a dependency parser using the English PennTreebank data. We implement the feed-forward neural network model of the dependency parser from the paper [_A Fast and Accurate Dependency Parser using Neural Networks_](https://nlp.stanford.edu/pubs/emnlp2014-depparser.pdf). The model is trained on the features obtained from the arc-standard parse of the labeled dependency trees.

### Arc-Standard Parse

The arc-standard parsing algorithm constructs the dependency parse tree in a bottom-up manner using shift and reduce operations. It uses two data structures, a stack and a buffer, to keep track of newly added dependency arcs. 

In the initial configuration, the stack contains a single node called ROOT and the buffer contains nodes corresponding to all the words of the sentence. 

At each step, the arc-standard algorithm performs one of three operations &mdash; *LEFT-ARC*, *RIGHT-ARC* or *SHIFT* &mdash; depending upon the dependency tree of the sentence. The operations change the buffer and stack contents, and adds new dependency edges between nodes of the stack.

The algorithm terminates when the buffer is empty and the stack contains the single ROOT node. The constructed dependency tree is rooted at the ROOT node in the stack.

### Features

At inference, the labeled dependency tree is not available and the transition operation &mdash; *LEFT-ARC*, *RIGHT-ARC* or *SHIFT* &mdash; is instead decided by a neural model. The features used by this neural model are calculated from the stack and buffer contents.

Let $s_i$ be the $i^{th}$ node of the stack from the top, $b_i$ be the $i^{th}$ node of the buffer, $lc_j(x)$ be the $j^{th}$ leftmost child of node $x$, and $rc_j(x)$ be the $j^{th}$ rightmost child of node $x$. For node $x$, let $x.w$ be the corresponding word of $x$, $x.t$ be the part-of-speech, and $x.l$ be the label of the dependency arc between $x$ and the parent of $x$.

As per the paper, the feature vector to the model is 48-dimensional and is calculated as follows &mdash;

1. features[1..6] = s<sub>1</sub>.w, s<sub>2</sub>.w, s<sub>3</sub>.w, b<sub>1</sub>.w, b<sub>2</sub>.w, b<sub>3</sub>.w
2. features[7..10] = lc<sub>1</sub>(s<sub>1</sub>).w, rc<sub>1</sub>(s<sub>1</sub>).w, lc<sub>2</sub>(s<sub>1</sub>).w, rc<sub>2</sub>(s<sub>1</sub>).w. 
features[11..14] are similarly calculated, but using s<sub>2</sub> instead of s<sub>i</sub>.
3. features[15,16] = lc<sub>1</sub>(lc<sub>1</sub>(s<sub>1</sub>)).w, rc<sub>1</sub>(rc<sub>1</sub>(s<sub>1</sub>)).w. 
features[17,18] are similarly calculated, but using s<sub>2</sub> instead of s<sub>1</sub>.
4. features[19..36] are calculated similarly as features [1..18], but we use the part-of-speech attribute instead of the word.
If features[i] = x.w, then features[i + 18] = x.t, for i = 1..18
5. features[37..48] are calculated similarly as features [7..18], but we use the dependency label attribute instead of the word.
If features[i] = x.w, then features[i + 30] = x.l, for i = 7..18

If the corresponding node for some feature is absent, for example, s<sub>2</sub>.w when the stack contains only one element, we use `null` value instead for the feature.

## Requirements

`
torch>=1.5, numpy>=1.18, pandas>=0.25, tqdm>=4.46, matplotlib>=3.1, pptree==3.1(optional)
`

We tested the code using python 3.6. `pptree` is used to draw the dependency tree on the terminal.

## Usage

The following steps show how to train the dependency parser model and test it on the development set.
 
 
1.  Run `python preparedata.py` to find the arc-standard parse of _train.orig.conll_ and _dev.orig.conll_ files. It creates five files &mdash;
    &nbsp;
    i.  *train.converted* and *dev.converted* contain text representations of the transition features, as described above.
    ii. *train.pt* and *dev.pt* contain torch tensors created from the aforementioned text feature files. These are used in training.
    iii. *vocab.dict* contains the word, part-of-speech and arc label vocabulary, and the label set.
&nbsp;

2.  Run `python train.py` to train the dependency parser model and create the _train.model_ file. 
    &nbsp;
    It also creates a plot of the training set and development set losses against the number of epochs in _loss.png_ file, and saves the loss values for each epoch in _loss.txt_ file.
    &nbsp;
    You can change the batch size and number of epochs using `-b` and `-e` arguments. The default values of batch size and number of epochs is 1024 and 1000 respectively. For quicker results, you can reduce the number of epochs. We recommend a value of 100.
    &nbsp;
    Set `-c` to use the cube activation function. 
    For gpu training, pass the cuda device index number through the `-d` argument. By default, training occurs on the cpu.
&nbsp;

3. Run `python parse.py -m train.model -i dev.orig.conll -o dev.predicted.conll` to parse the sentences in _dev.orig.conll_ and create the _dev.predicted.conll_ file, containing the predicted dependency parse tree labels. 
    &nbsp;
    You can use your own conll file, which you need to parse, instead of _dev.orig.conll_. 
    Set the `-c` option if _train.model_ uses cube activation function.
&nbsp;

4. Run `python evaluate.py -t dev.orig.conll -p dev.predicted.conll` to evaluate the predicted labels in _dev.predicted.conll_ file against the true labels in _dev.orig.conll_ file. It will output two scores &mdash; unlabeled attachment score (UAS) and labeled attachment score (LAS). The UAS is the accuracy in predicting the parent of each word in the dependency tree, and LAS is the accuracy in predicting the correct parent of each word, and the dependency label of the edge connecting the word to its parent in the dependency tree.
    &nbsp;
    On the given development set, you should get a UAS score of around **0.90** and LAS score of around **0.89**, using the default values of batch size and number of epochs.

## Files

1. `tree.py` defines the tree data structure used to represent labeled and partially labeled dependency trees.
2. `stack.py` defines the stack data structure used to create the stack and buffer of the arc-standard algorithm.
3. `check_projective.py` checks if a set of arcs is projective.
4. `transition_feature.py` finds the features of a single arc-standard parse configuration, as described above, from its stack and buffer.
5. `preparedata.py` finds the arc-standard parse features and vocabularies for training the neural dependency parser.
6. `train.py` trains the model using the features, outputed by `preparedata.py`.
7. `parse.py` parses an input conll file and outputs a labeled conll file, using the dependency parser trained by `train.py`.
8. `evaluate.py` outputs the UAS and LAS scores for a predicted labeled conll file, outputted by `parse.py`, against the conll file containing the true labels.

## Reference

Chen, Danqi, and Christopher D. Manning. "A fast and accurate dependency parser using neural networks." Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP). 2014.