# author - Sabyasachee

class Stack:
    '''
    stack data structure for Stack and Buffer in arc-standard parse
    '''

    def __init__(self, maxlen=150):
        self.items = [None for _ in range(maxlen)]
        self.top = -1

    def push(self, item):
        self.top += 1
        self.items[self.top] = item

    def __getitem__(self, index):
        return self.items[self.top - index + 1]

    def pop(self):
        self.top -= 1

    def pop2(self):
        self.items[self.top - 1] = self.items[self.top]
        self.top -= 1

    def __len__(self):
        return self.top + 1

    def __repr__(self):
        return " --- ".join([str(x) for x in self.items[:self.top + 1]])

    def __format__(self, spec):
        if spec=="r":
            return " --- ".join([str(x) for x in reversed(self.items[:self.top + 1])])
        return " --- ".join([str(x) for x in self.items[:self.top + 1]])