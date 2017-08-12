#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import pyximport; pyximport.install()
from cdparser.calgorithm import *


class Node(object):

    def __init__(self, word, pos, norm=None, lemma=None, features=None):
        self.word = word
        self.pos = pos
        if norm:
            self.norm = norm
        else:
            self.norm = word

    def __repr__(self):
        return "%s-%s" % (self.word, self.pos)


class DependencyGraph(object):

    def __init__(self, nodes):
        self.nodes = np.array([Node("*ROOT*", "*ROOT*")] + list(nodes))
        self.heads = np.array([-1] * len(self.nodes))
        self.rels = np.array([None] * len(self.nodes))

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.nodes = self.nodes
        result.heads = self.heads.copy()
        result.rels = self.rels.copy()
        return result

    def cleaned(self):
        return DependencyGraph(self.nodes[1:])

    def attach(self, head, tail, rel):
        self.heads[tail] = head
        self.rels[tail] = rel

    def intersection(self, graph):
        newgraph = self.cleaned()
        for i in range(len(self.nodes)):
            if self.heads[i] == graph.heads[i]:
                newgraph.heads[i] = self.heads[i]
            if self.rels[i] == graph.rels[i]:
                newgraph.rels[i] = self.rels[i]
        return newgraph

    def isCompatible(self, head, tail):
        if self.heads[tail] >= 0:
            return head == self.heads[tail]

        self.heads[tail] = head
        res = self.isProjective()
        self.heads[tail] = -1
        return res

    def isProjective(self):
        """ Check whether this graph is projective

        Complexity: O(n^2) -- check every pair of edges
        """
        return cIsProjective(self.heads)

    def leftChild(self, head, num=0):
        if head < 0:
            return -1
        cur = 0
        for i in range(0, head):
            if self.heads[i] == head:
                if cur == num:
                    return i
                else:
                    cur += 1
        return -1

    def rightChild(self, head, num=0):
        if head < 0:
            return -1
        cur = 0
        for i in range(len(self.nodes) - 1, head, -1):
            if self.heads[i] == head:
                if cur == num:
                    return i
                else:
                    cur += 1
        return -1

    def getWord(self, i):
        return 0 if i < 0 else self.nodes[i].word

    def getPos(self, i):
        return 0 if i < 0 else self.nodes[i].pos

    def __repr__(self):
        return "\n".join(["{} ->({})  {} ({})".format(str(self.nodes[i]), self.rels[i], self.heads[i], self.nodes[self.heads[i]]) for i in range(len(self.nodes))])
