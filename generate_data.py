import argparse
import sys
import pickle as pkl
import json
import pygtrie as trie
import string

def parse_json_like(fname):
    with open(fname) as f:
        for line in f.readlines():
            js = json.loads(line)
            if "queryInImp" not in js or "count" not in js: continue
            cnt = int(js["count"])
            if cnt < 51: continue #only keep results with 5 searches
            s = js["queryInImp"]
            if all(ord(_c) < 128 for _c in s):
                yield "/".join(filter(lambda s: s.strip() != "", s.lower().split())), int(js["count"])
            else: continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform some initial testing for question intention model training")
    parser.add_argument("--data", default="./data.json", help="Path of training data in json (scattered)\
     format")
    # parser.add_argument("--model", default="./models", help="Directory to output of models")
    # parser.add_argument("--vocab-size", default=20000, type=int, help="Tokenizer vocabulary size")
    # parser.add_argument("--tokenizer-mode", default="count", help="Tokenizer vectorization method")
    # parser.add_argument("--batch", default=500, type=int, help="minibatch batch size")
    # parser.add_argument("--epoch", default=1, type=int, help="Number of epochs")
    # parser.add_argument("--debug", default=False, action="store_true", help="Output debugging info")
    # parser.add_argument("--mode", default="lr", help="""Mode of action:
    # lr -- tensorflow linear classifiers, logistic regression
    # lr2 -- tensorflow linear classifiers, logistic regression with weight decay + ftrl
    # nn -- keras nn, 1 hidden layer, 1 dropout
    # nn2 -- keras nn, 2 hidden layers, 1 dropout
    # lstmcnn -- keras lstm cnn with trained embedding layer
    # """)
    args = parser.parse_args()
    tt = trie.StringTrie(sep="/")
    cnt = 0 
    for q, c in parse_json_like(args.data):
        if q in tt:
            try:
                tt[q] += c 
            except:
                print("error setting " + tt[q])
                tt[q] = c
        else:
            tt[q] = c
        cnt += 1
        if cnt % 10000 == 0:
            print(cnt, len(tt)) 
    pkl.dump(tt, open("mytrie.gtrie", "wb"))