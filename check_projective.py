# author - Sabyasachee

def is_non_projective(arcs, verbose=False):
    '''
    return true if arcs are non-projective, else return false
    arcs is a list of 2-tuples of integers. 
    Each tuple (x, y) is a dependency arc between token idx x and y, and x < y.
    '''
    for i in range(len(arcs)):
        for j in range(i + 1, len(arcs)):
            x, y = arcs[i]
            a, b = arcs[j]
            if (x < a and a < y and y < b) or (a < x and x < b and b < y):
                if verbose:
                    print(f"({x},{y}), ({a},{b}) intersect")
                return True
    return False