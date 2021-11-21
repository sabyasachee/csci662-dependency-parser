# author - Sabyasachee
import argparse
import torch
import pickle
from tqdm import tqdm

from tree import Tree
from stack import Stack
from transition_feature import create_transition_feature
from dependency_parser import DependencyParser

def parse(model, vocab, example):
    '''
    parse the example given in conll format using model and vocab.
    return the parsed tree in conll format and as data structure
    '''
    # read the example conll
    data = []
    for line in example.split("\n"):
        values = line.split()
        data.append(values)

    # read the vocab dictionaries
    word_vocab_to_id = vocab["word"]
    pos_vocab_to_id = vocab["pos"]
    arc_vocab_to_id = vocab["arc"]
    label_to_id = vocab["label"]
    label_set = ["" for _ in range(len(label_to_id))]
    for label, _id in label_to_id.items():
        label_set[_id] = label
        
    # create an unparsed tree
    tree = Tree(example, parsed=False)

    # initialize stack and buffer
    n_words = len(tree.nodes) - 1
    stack = Stack(n_words + 1)
    buffer = Stack(n_words)
    
    stack.push((tree.nodes[0],))
    for node in reversed(tree.nodes[1:]):
        buffer.push((node,))
        
    # loop until stack has only one element, ROOT, and buffer is empty
    while len(stack) > 1 or len(buffer):
        
        # create feature from stack and buffer
        feature = create_transition_feature(stack, buffer)
        
        # convert to id using vocab
        for i in range(18):
            if feature[i] in word_vocab_to_id:
                feature[i] = word_vocab_to_id[feature[i]]
            else:
                feature[i] = word_vocab_to_id["UNK"]
        
        for i in range(18,36):
            feature[i] = pos_vocab_to_id[feature[i]]
        
        for i in range(36,48):
            feature[i] = arc_vocab_to_id[feature[i]]
            
        # pass through model
        inp = torch.LongTensor(feature).reshape(1,-1)
        out = model(inp)
        label_id = out.argmax().item()
        label2_id = out.argsort().flatten()[-2].item()
        label = label_set[label_id]
        label2 = label_set[label2_id]
        
        # find transition
        if len(stack) == 1:
            transition = "SHIFT"
        elif len(buffer) == 0:
            if label != "SHIFT":
                transition = label
            else:
                transition = label2
        else:
            transition = label
        
        # make changes to stack and buffer according to transition
        if transition == "SHIFT":
            stack.push(buffer[1])
            buffer.pop()
        else:
            direction, arc = transition.split("-")
            
            if direction == "LEFT":
                tree.connect(stack[1][0], stack[2][0], arc)
                stack.pop2()
                
            else:
                tree.connect(stack[2][0], stack[1][0], arc)
                stack.pop()

    # write predicted parse tree in conll format 
    for node in tree.nodes[1:]:
        parent = node.parent
        node_index = node.index
        parent_index = parent.index
        arc = node.arc_label
        data[node_index-1][6] = str(parent_index)
        data[node_index-1][7] = arc
        
    # return conll tree and data structure tree
    conll = "\n".join("\t".join(rec) for rec in data)
    return conll, tree

def parse_conll(model_file, conll_file, vocab_file, output_conll_file, use_cube_activation):
    # read vocab
    print("reading vocabulary")
    vocab = pickle.load(open(vocab_file, "rb"))
    
    # load model
    print("loading model")
    word_vocab_size = len(vocab["word"])
    pos_tag_size = len(vocab["pos"])
    arc_tag_size = len(vocab["arc"])
    label_size = len(vocab["label"])

    model = DependencyParser(word_vocab_size, pos_tag_size, arc_tag_size, label_size, use_cube_activation=use_cube_activation)
    model.load_state_dict(torch.load(model_file, map_location="cpu"))
    model.eval()

    # read conll
    conll = open(conll_file).read().strip()
    examples = conll.split("\n\n")

    # failed indices are the indices of the sentences that the model failed to parse
    predicted_examples = []
    failed_indices = []

    # parse the examples in the conll file
    for i, example in tqdm(enumerate(examples), total=len(examples), desc="parsing"):
        try:
            predicted_example, _ = parse(model, vocab, example)
        except Exception:
            failed_indices.append(i)
            predicted_example = example
        predicted_examples.append(predicted_example)
    
    print(f"failed to parse {len(failed_indices)} sentences")

    # write the parsed conll
    print("writing output")
    predicted_conll = "\n\n".join(predicted_examples)
    open(output_conll_file,"w").write(predicted_conll)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="dependency parser")
    
    parser.add_argument("-m", type=str, help="model file", default="train.model")
    parser.add_argument("-c", action="store_true", help="set if model uses cube activation function")
    parser.add_argument("-i", type=str, help="input conll file")
    parser.add_argument("-o", type=str, help="output conll file")
    parser.add_argument("-d", type=str, help="vocabulary file", default="vocab.dict")

    args = parser.parse_args()
    parse_conll(args.m, args.i, args.d, args.o, args.c)