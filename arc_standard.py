# author - Sabyasachee

import numpy as np
from stack import Stack
from transition_feature import create_transition_feature

def fill_descendant(node, rightmost_descendant):
    '''
    find the rightmost descendant (rn) of the words in the sentence in the dependency tree.
    for word i, if word j_1, j_2, ..., j_n are its children, then rn[i] = max(j_1, j_2, ..., j_n, rn(j_1), rn(j_2), ..., rn(j_n))
    '''
    i = node.index
    rightmost_descendant[i] = i

    for child in node.children:
        fill_descendant(child, rightmost_descendant)
        j = child.index
        rightmost_descendant[i] = max(rightmost_descendant[i], rightmost_descendant[j])

def create_features(parsed_tree, unparsed_tree, verbose=False):
    '''
    find the transitions in the arc-standard parse of the unparsed_tree, and create features of each transition.
    the parsed_tree is used to find the transition at each step.
    return 2-dimensional Nx48 feature list and N-length label list, where N is the total number of transitions.
    '''

    # initialize stack and buffer
    # stack and buffer items are 2-tuples of the nodes of the unparsed and parsed tree
    n_words = len(parsed_tree.nodes) - 1
    stack = Stack(n_words + 1)
    buffer = Stack(n_words)

    stack.push((unparsed_tree.nodes[0], parsed_tree.nodes[0]))
    for node1, node2 in zip(reversed(unparsed_tree.nodes[1:]), reversed(parsed_tree.nodes[1:])):
        buffer.push((node1,node2))

    # find the rightmost descendant of each tree node. this is used in the right arc
    rightmost_descendant = np.arange(len(parsed_tree.nodes))
    fill_descendant(parsed_tree.nodes[0], rightmost_descendant)

    features = []
    labels = []
    
    # loop until buffer is empty and stack contains only root (stack has 1 element)
    while len(stack) > 1 or len(buffer):
        if verbose:
            print(f"stack:{len(stack)}  = {stack}")
            print(f"buffer:{len(buffer)} = {buffer:r}")

        # create transition feature
        feature = create_transition_feature(stack, buffer)

        # if len(stack) >= 2 and top element of stack is parent of second element of stack, then LEFT-ARC
        if len(stack) >= 2 and stack[2][1].parent == stack[1][1]:
            label = f"LEFT-{stack[2][1].arc_label}"
            unparsed_tree.connect(stack[1][0], stack[2][0], stack[2][1].arc_label)
            stack.pop2()

        # if len(stack) >= 2, second element of stack is parent of top element of stack, and no descendant of top element is in buffer, then RIGHT-ARC
        elif len(stack) >= 2 and stack[1][1].parent == stack[2][1] and (len(buffer) == 0 or rightmost_descendant[stack[1][1].index] < buffer[1][1].index):
            label = f"RIGHT-{stack[1][1].arc_label}"
            unparsed_tree.connect(stack[2][0], stack[1][0], stack[1][1].arc_label)
            stack.pop()

        # else SHIFT
        else:
            stack.push(buffer[1])
            buffer.pop()
            label = "SHIFT"

        if verbose:
            print(f"{label}\n")

        features.append(feature)
        labels.append(label)

    return features, labels