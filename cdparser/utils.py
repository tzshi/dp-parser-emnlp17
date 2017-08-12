#!/usr/bin/env python
# encoding: utf-8

from cdparser.graph import DependencyGraph, Node
from collections import Counter
import re
import os


def normalize(word):
    return re.sub(r"\d", "0", word).lower()


def buildVocab(graphs, cutoff=0):
    wordsCount = Counter()
    posCount = Counter()
    relCount = Counter()

    for graph in graphs:
        wordsCount.update([node.norm for node in graph.nodes[1:]])
        posCount.update([node.pos for node in graph.nodes[1:]])
        relCount.update(graph.rels[1:])

    # In case some arcs have no labels
    del relCount[None]

    wordsCount = Counter({w: i for w, i in wordsCount.items() if i > cutoff})
    print("Vocab containing {} words".format(len(wordsCount)))

    ret = {
        "vocab": list(wordsCount.keys()),
        "freq": wordsCount,
        "pos": list(posCount.keys()),
        "rels": list(relCount.keys())
    }

    return ret


def readConll(f):
    # Get a single graph
    def getGraph(nodes, edges):
        graph = DependencyGraph(nodes)
        for (head, tail, rel) in edges:
            graph.attach(head, tail, rel)
        return graph

    nodes = []
    edges = []
    for l in f:
        l = l.strip()
        if len(l) > 0 and l[0] == "#":
            continue
        if len(l) == 0:
            if len(nodes) > 0:
                yield getGraph(nodes, edges)
                nodes = []
                edges = []
        else:
            [id, form, lemma, cpostag, postag, feats, head, deprel, phead, pdeprel] = l.split()
            if head == "_":
                continue
            nodes.append(Node(form, postag, normalize(form)))
            edges.append((int(head), int(id), deprel))
    # If there is no "\n" in the final line
    if len(nodes) > 0:
        yield getGraph(nodes, edges)


def writeConll(graphs, f):
    for graph in graphs:
        for i in range(1, len(graph.nodes)):
            # Assume that we only need pos, but not both cpos and pos
            f.write("{}\t{}\t_\t{}\t{}\t_\t{}\t{}\t_\t_\n".format(i, graph.nodes[i].word, graph.nodes[i].pos, graph.nodes[i].pos, graph.heads[i], graph.rels[i]))
        f.write("\n")


def scriptResultNumbers(fname):
    with open(fname, "r") as f:
        las = float(f.readline().strip()[-7:-2])
        uas = float(f.readline().strip()[-7:-2])
        return (las, uas)


def scriptEvaluate(system_graph, gold_graph):
    with open("__tmpsystemgraph", "w") as f:
        writeConll(system_graph, f)
    with open("__tmpgoldgraph", "w") as f:
        writeConll(gold_graph, f)
    os.system('perl utils/eval.pl -g __tmpgoldgraph -s __tmpsystemgraph > __tmpevalresult')
    return scriptResultNumbers("__tmpevalresult")
