# author - Sabyasachee

try:
    from pptree import print_tree
except ImportError:
    pass

class TreeNode:
    
    def __init__(self, line=None, parsed=True):
        '''
        node of a dependency tree, created from a line in conll file.
        if line is None, a root node is created.
        if parsed is False, parent_index and arc_label are set to default values
        '''
        if line is not None:
            values = line.split()
            self.index = int(values[0])
            self.word = values[1]
            self.lemma = values[2]
            self.coarse_POS = values[3]
            self.fine_POS = values[4]
            if parsed:
                self.parent_index = int(values[6])
                self.arc_label = values[7]
            else:
                self.parent_index = -1
                self.arc_label = "-"
        else:
            self.index = 0
            self.word = "ROOT"
            self.lemma = "ROOT"
            self.coarse_POS = "ROOTPOS"
            self.fine_POS = "ROOTPOS"
            self.parent_index = -1
            self.arc_label = "-"

        self.parent = None
        self.children = []

    def __repr__(self):
        return f"{self.index}:{self.word}"

class Tree:

    def __init__(self, example, parsed=True):
        '''
        dependency tree created from text in conll format.
        if parsed is False, nodes are not connected with each other.
        '''
        lines = example.split("\n")
        self.nodes = [TreeNode()]

        for line in lines:
            node = TreeNode(line, parsed=parsed)
            self.nodes.append(node)

        if parsed:
            for i in range(1,len(self.nodes)):
                node = self.nodes[i]
                parent_node = self.nodes[node.parent_index]
                node.parent = parent_node
                parent_node.children.append(node)

            for i in range(len(self.nodes)):
                node = self.nodes[i]
                node.children = sorted(node.children, key=lambda child_node: child_node.index)

    def print_tree(self):
        try:
            print_tree(self.nodes[0])
        except:
            for node in self.nodes:
                print(f"{node} -> {node.children}")
        
    def connect(self, parent_node, child_node, arc_label):
        parent_node.children.append(child_node)
        child_node.parent = parent_node
        child_node.arc_label = arc_label