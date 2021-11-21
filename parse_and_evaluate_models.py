from threading import main_thread
from parse import parse_conll
from evaluate import evaluate

def parse_and_evaluate(model_file, use_cube_activation):
    input_conll_file = "dev.orig.conll"
    output_conll_file = "dev.predicted.conll"
    vocab_file = "vocab.dict"
    parse_conll(model_file, input_conll_file, vocab_file, output_conll_file, use_cube_activation)
    return evaluate(input_conll_file, output_conll_file)

if __name__ == "__main__":
    with open("scores.txt","w") as fw:
        fw.write("model\tuas\tlas\n")
        for use_cube_activation in [True, False]:
            for i in range(1,11):
                model_file = f"train_cube-{use_cube_activation}_1024_{100*i}.model"
                uas, las = parse_and_evaluate(model_file, use_cube_activation)
                fw.write(f"{model_file}\t{uas:.5f}\t{las:.5f}")