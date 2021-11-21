# author - Sabyasachee

from tqdm import tqdm, trange
import numpy as np
import pandas as pd
import torch
import pickle

from check_projective import is_non_projective
from tree import Tree
from arc_standard import create_features

def preparedata():
    '''
    convert train.orig.conll and dev.orig.conll to text features and tensor arrays, and save it.
    find the word, pos tag, arc tag and label vocab to index mappings and save it.
    '''

    # read conll files
    train_conll = open("train.orig.conll").read().strip()
    dev_conll = open("dev.orig.conll").read().strip()

    train_examples = train_conll.split("\n\n")
    dev_examples = dev_conll.split("\n\n")

    print(f"{len(train_examples)} train examples")
    print(f"{len(dev_examples)} dev examples")

    # find the indices of the non-projective trees, word vocab, pos tag vocab, arc vocab and label set
    non_projective_indices = [[],[]]
    word_vocab = set()
    pos_vocab = set()
    arc_vocab = set()

    for k, examples in enumerate([train_examples, dev_examples]):
        for l, example in tqdm(enumerate(examples), total=len(examples), desc="finding non-projective trees"):

            arcs = []
            for line in example.split("\n"):
                values = line.split()

                if k == 0:
                    word_vocab.add(values[1])
                    pos_vocab.add(values[3])
                    arc_vocab.add(values[7])

                x, y = int(values[0]), int(values[6])
                if x < y:
                    arcs.append((x, y))
                else:
                    arcs.append((y, x))

            if is_non_projective(arcs):
                non_projective_indices[k].append(l)

    print("\n")
    print(f"{len(non_projective_indices[0])} non-projective trees in train set. %projective = {1 - len(non_projective_indices[0])/len(train_examples):.4f}")
    print(f"{len(non_projective_indices[1])} non-projective trees in devel set. %projective = {1 - len(non_projective_indices[1])/len(dev_examples):.4f}")

    word_vocab.update(["ROOT","NULLWORD","UNK"])
    pos_vocab.update(["ROOTPOS","NULLPOS"])
    label_set = set([f"{direction}-{arc}" for arc in arc_vocab for direction in ["LEFT","RIGHT"]] + ["SHIFT"])
    arc_vocab.add("NULLARC")

    print("\n")
    print("vocab lengths =>")
    print(f"\tword: {len(word_vocab)}")
    print(f"\tpos: {len(pos_vocab)}")
    print(f"\tarc: = {len(arc_vocab)}")
    print(f"number of labels = {len(label_set)}")

    word_vocab_to_id = dict([(word, i) for i, word in enumerate(word_vocab)])
    pos_vocab_to_id = dict([(pos, i) for i, pos in enumerate(pos_vocab)])
    arc_vocab_to_id = dict([(arc, i) for i, arc in enumerate(arc_vocab)])

    label_set = sorted(label_set)
    label_to_id = dict([(label, i) for i, label in enumerate(label_set)])

    vocab = {"word":word_vocab_to_id, "pos":pos_vocab_to_id, "arc":arc_vocab_to_id, "label":label_to_id}

    # create features from the arc-standard parse
    features = [[],[]]
    labels = [[],[]]

    for k, examples in enumerate([train_examples, dev_examples]):
        for l, example in tqdm(enumerate(examples), total=len(examples), desc="arc standard"):
            
            # arc-standard cannot handle non-projective trees
            if l not in non_projective_indices[k]:
                
                # create the trees
                parsed_tree = Tree(example)
                unparsed_tree = Tree(example, parsed=False)

                # arc standard parse
                feature, label = create_features(parsed_tree, unparsed_tree)
                features[k] += feature
                labels[k] += label

    print("\n")
    print(f"#Train = {len(features[0])}")
    print(f"#Devel = {len(features[1])}")

    # create dataframe and save
    print("\n")
    print("creating text feature files")
    header = ["s1.w","s2.w","s3.w","b1.w","b2.w","b3.w","lc1(s1).w","rc1(s1).w","lc2(s1).w","rc2(s1).w","lc1(s2).w","rc1(s2).w","lc2(s2).w","rc2(s2).w","lc1(lc1(s1)).w","rc1(rc1(s1)).w","lc1(lc1(s2)).w","rc1(rc1(s2)).w","s1.t","s2.t","s3.t","b1.t","b2.t","b3.t","lc1(s1).t","rc1(s1).t","lc2(s1).t","rc2(s1).t","lc1(s2).t","rc1(s2).t","lc2(s2).t","rc2(s2).t","lc1(lc1(s1)).t","rc1(rc1(s1)).t","lc1(lc1(s2)).t","rc1(rc1(s2)).t","lc1(s1).l","rc1(s1).l","lc2(s1).l","rc2(s1).l","lc1(s2).l","rc1(s2).l","lc2(s2).l","rc2(s2).l","lc1(lc1(s1)).l","rc1(rc1(s1)).l","lc1(lc1(s2)).l","rc1(rc1(s2)).l"]

    train = pd.DataFrame(data=features[0], columns=header)
    devel = pd.DataFrame(data=features[1], columns=header)
    train["label"] = labels[0]
    devel["label"] = labels[1]

    print("saving text feature files")
    train.to_csv("train.converted", sep="\t", index=False)
    devel.to_csv("dev.converted", sep="\t", index=False)

    # handle unknown words in dev set
    devel_word_data = devel.iloc[:,:18].to_numpy()
    m, n = devel_word_data.shape

    print("\n")
    print("replacing unknown words with UNK")
    for i in trange(m):
        for j in range(n):
            if devel_word_data[i,j] not in word_vocab:
                devel_word_data[i,j] = "UNK"
    devel.iloc[:,:18] = devel_word_data

    # convert to tensor
    train_data = train.to_numpy()
    devel_data = devel.to_numpy()

    print("\n")
    print("convert text features to tensor")
    for data in [train_data, devel_data]:    
        m, n  = data.shape

        for i in trange(m):
            for j in range(n):

                if j < 18:
                    data[i,j] = word_vocab_to_id[data[i,j]]
                elif j < 36:
                    data[i,j] = pos_vocab_to_id[data[i,j]]
                elif j < 48:
                    data[i,j] = arc_vocab_to_id[data[i,j]]
                else:
                    data[i,j] = label_to_id[data[i,j]]

    train_tensor = torch.from_numpy(train_data.astype(np.int)).long()
    devel_tensor = torch.from_numpy(devel_data.astype(np.int)).long()

    # save tensor and label, word, pos tag and arc to index dictionaries

    print("\n")
    print("saving tensor and vocab to id")
    torch.save(train_tensor, "train.pt")
    torch.save(devel_tensor, "dev.pt")
    pickle.dump(vocab, open("vocab.dict","wb"))

if __name__ == "__main__":
    preparedata()