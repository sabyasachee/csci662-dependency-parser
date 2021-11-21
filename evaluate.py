# author - Sabyasachee
import argparse

def evaluate(true_conll_file, pred_conll_file):
    '''
    print unlabeled and labeled attachment scores of pred_conll_file against true_conll_file.
    '''
    true_conll = open(true_conll_file).read().strip()
    pred_conll = open(pred_conll_file).read().strip()

    true_examples = true_conll.split("\n\n")
    pred_examples = pred_conll.split("\n\n")

    unlabeled_correct = 0
    labeled_correct = 0
    total = 0

    for true_example, pred_example in zip(true_examples, pred_examples):
        for true_line, pred_line in zip(true_example.split("\n"), pred_example.split("\n")):
            
            true_values, pred_values = true_line.split(), pred_line.split()
            
            if true_values[6] == pred_values[6]:
                unlabeled_correct += 1
                
                if true_values[7] == pred_values[7]:
                    labeled_correct += 1
                    
            total += 1
            
    uas, las = unlabeled_correct/total, labeled_correct/total
    print(f"unlabeled attachment score = {uas:.4f}")
    print(f"labeled attachment score = {las:.4f}")

    return uas, las

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="evaluate output of dependency parser")
    parser.add_argument("-t", type=str, help="conll file containing true labels")
    parser.add_argument("-p", type=str, help="conll file containing output of dependency parser")

    args = parser.parse_args()
    evaluate(args.t, args.p)