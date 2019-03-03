import argparse
import sys
import pickle as pkl
import json
import pygtrie as trie
import string
import os
import random


def parse_json_like(fname):
    with open(fname, encoding="utf-8") as f:
        for line in f.readlines():
            js = json.loads(line)
            if "queryInImp" not in js or "count" not in js:
                continue
            cnt = int(js["count"])
            if cnt < 100:
                continue  # only keep results with 10 searches
            s = js["queryInImp"]
            if all(ord(_c) < 128 for _c in s) and s.strip() != "":
                yield "/".join(filter(lambda s: s.strip() != "", s.lower().split())), int(js["count"])
            else:
                continue


def get_example(allow_single=True, pft="./mytrie.gtrie"):
    """
    generate a training sample in the form
    query (last word trimmed if length > 1),
    correct query, count
    candidate 1, count etc

    allow_single to allow single word queries
    """
    searched_queries = []
    if os.path.exists(pft):
        tt = pkl.load(open(pft, "rb"))
        print("Loaded trie at {}".format(pft))
    else:
        raise IOError("Generate prefix trie first")
    for query, query_cnt in tt.iteritems():
        query = query.split("/")
        if len(query) < 2:
            if not allow_single:
                continue
            trimmed_query = query
        else:
            trimmed_query = query[:-1]
        trimmed_query = "/".join(trimmed_query)
        query = "/".join(query)
        if trimmed_query in searched_queries:
            continue
        searched_queries.append(trimmed_query)
        outputs = []
        for q, c in tt.iteritems(prefix=trimmed_query):
            outputs.append((q, c))
        if len(outputs) < 2:
            continue
        outputs = sorted(outputs, key=lambda x: -x[1])[:9]
        if query not in [i[0] for i in outputs]:
            outputs.append((query, query_cnt))
        random.shuffle(outputs)
        yield trimmed_query, (query, query_cnt), outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform some initial testing for question intention model training")
    parser.add_argument("--data", default="./data.json", help="Path of training data in json (scattered)\
     format")
    parser.add_argument("--trie", default="./mytrie.gtrie",
                        help="Path of processed prefix trie")
    # parser.add_argument("--model", default="./models", help="Directory to output of models")
    # parser.add_argument("--vocab-size", default=20000, type=int, help="Tokenizer vocabulary size")
    # parser.add_argument("--tokenizer-mode", default="count", help="Tokenizer vectorization method")
    # parser.add_argument("--batch", default=500, type=int, help="minibatch batch size")
    # parser.add_argument("--epoch", default=1, type=int, help="Number of epochs")
    parser.add_argument("--debug", default=False,
                        action="store_true", help="Output debugging info")
    # parser.add_argument("--mode", default="lr", help="""Mode of action:
    # lr -- tensorflow linear classifiers, logistic regression
    # lr2 -- tensorflow linear classifiers, logistic regression with weight decay + ftrl
    # nn -- keras nn, 1 hidden layer, 1 dropout
    # nn2 -- keras nn, 2 hidden layers, 1 dropout
    # lstmcnn -- keras lstm cnn with trained embedding layer
    # """)
    args = parser.parse_args()
    tt = trie.StringTrie()
    if os.path.exists(args.trie):
        tt = pkl.load(open(args.trie, "rb"))
        print("Loaded trie at {}".format(args.trie))
    else:
        cnt = 0
        for q, c in parse_json_like(args.data):
            if q in tt:
                try:
                    tt[q] += c
                except:
                    print("error setting {} to with val {}".format(q, tt[q]))
                    tt[q] = c
            else:
                tt[q] = c
            cnt += 1
            if cnt % 10000 == 0:
                print(cnt, len(tt))
        pkl.dump(tt, open(args, trie, "wb"))
    if args.debug:
        for items in tt.items(prefix="linkedin"):
            print(items)
        cnt = 0
        for tq, cq, ots in get_example():
            print(tq, cq, ots)
            cnt += 1
            if cnt > 10:
                break
        cnt = 0
        for tq, cq, ots in get_example(False):
            print(tq, cq, ots)
            cnt += 1
            if cnt > 10:
                break
