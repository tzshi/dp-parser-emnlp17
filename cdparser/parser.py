#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function
import time, random
random.seed(1)
import os, os.path
from dynet import *
from operator import itemgetter
import numpy as np
from cdparser.layers import Dense, MultiLayerPerceptron, Bilinear, identity

import pyximport; pyximport.install()
from cdparser.calgorithm import *
from cdparser.aedpalgorithm import *
from cdparser.ahdpalgorithm import *
from cdparser.ahbeamconfiguration import AHBeamConfiguration
from cdparser.aebeamconfiguration import AEBeamConfiguration


class ComputationCarrier(object):

    def __copy__(self):
        result = object.__new__(ComputationCarrier)
        result.__dict__.update(self.__dict__)
        return result


class Logger(object):
    pass


class CDParser:

    def __init__(self, vocab, freq, pos, rels, **kwargs):
        self.model = Model()
        self.learning_rate = kwargs.get("learningRate", 0.001)
        self.beta2 = kwargs.get("beta2", 0.999)
        self.edecay = kwargs.get("edecay", 0.)
        self.clip = kwargs.get("clip", 5.)
        self.sparse_updates = kwargs.get("sparse_updates", True)
        self.dlm = kwargs.get("dlm", False)

        self.trainer = AdamTrainer(self.model, alpha=self.learning_rate, beta_2=self.beta2, edecay=self.edecay)
        self.trainer.set_sparse_updates(self.sparse_updates)
        self.trainer.set_clip_threshold(self.clip)

        self.activations = {'tanh': tanh, 'sigmoid': logistic, 'relu': rectify, 'tanh3': (lambda x: tanh(cwise_multiply(cwise_multiply(x, x), x)))}
        self.batch_size = kwargs.get("batch_size", 1)

        # Basic feature extractors
        self.bildims = kwargs.get("bilstm_dims", 128)
        self.wdims = kwargs.get("wembedding_dims", 128)
        self.pdims = kwargs.get("pembedding_dims", 32)
        self.rdims = kwargs.get("rembedding_dims", 32)
        self.bilstm_layers = kwargs.get("bilstm_layers", 2)
        self.bilstm_dropout = kwargs.get("bilstm_dropout", 0.0)

        # Graph-based parser
        self.graph_mlpactivation = self.activations[kwargs.get('graph_mlpactivation', 'relu')]
        self.graph_mlpdim = kwargs.get("graph_mlpdim", 100)
        self.graph_mlplayers = kwargs.get("graph_mlplayers", 1)
        self.graph_mlpdropout = kwargs.get("graph_mlpdropout", 0.0)
        self.graph_method = kwargs.get("graph_method", "discrim")

        # Labeler
        self.labeler_mlpactivation = self.activations[kwargs.get('labeler_mlpactivation', 'relu')]
        self.labeler_mlpdim = kwargs.get("labeler_mlpdim", 128)
        self.labeler_mlplayers = kwargs.get("labeler_mlplayers", 1)
        self.labeler_mlpdropout = kwargs.get("labler_mlpdropout", 0.0)
        self.labeler_concatdim = kwargs.get("labeler_concatdim", 128)
        self.labeler_concatlayers = kwargs.get("labeler_concatlayers", 1)
        self.labeler_concatdropout = kwargs.get("labler_concatdropout", 0.0)

        # Arc Hybrid
        self.ah_mlpactivation = self.activations[kwargs.get('ah_mlpactivation', 'tanh')]
        self.ah_mlpdim = kwargs.get("ah_mlpdim", 100)
        self.ah_mlplayers = kwargs.get("ah_mlplayers", 1)
        self.ah_mlpdropout = kwargs.get("ah_mlpdropout", 0.0)
        self.ah_bilinearFlag = kwargs.get("ah_bilinear", False)
        self.root_first = kwargs.get("root_first", False)
        self.stack_features = kwargs.get("stack_features", 3)
        self.buffer_features = kwargs.get("buffer_features", 1)
        self.ah_nonpos = kwargs.get("ah_nonpos", False)

        # Arc Sibling
        self.as_mlpactivation = self.activations[kwargs.get('as_mlpactivation', 'tanh')]
        self.as_mlpdim = kwargs.get("as_mlpdim", 100)
        self.as_mlplayers = kwargs.get("as_mlplayers", 1)

        # Arc Eager
        self.ae_mlpactivation = self.activations[kwargs.get('ae_mlpactivation', 'tanh')]
        self.ae_mlpdim = kwargs.get("ae_mlpdim", 100)
        self.ae_mlplayers = kwargs.get("ae_mlplayers", 1)
        self.ae_mlpdropout = kwargs.get("ae_mlpdropout", 0.0)

        # Dynamic Oracle
        self.dynamic_oracle = kwargs.get("dynamic_oracle", True)
        self.dynamic_explore_rate = kwargs.get("dynamic_explore_rate", 0.9)

        # Vocabulary
        self.freq = freq
        self.vocab = {w: i + 3 for i, w in enumerate(vocab)}
        self.pos = {p: i + 3 for i, p in enumerate(pos)}
        self.rels = {r: i for i, r in enumerate(rels)}
        self.irels = rels

        self.vocab['*PAD*'] = 1
        self.pos['*PAD*'] = 1

        self.vocab['*ROOT*'] = 2
        self.pos['*ROOT*'] = 2

        self.logger = Logger()
        self.logger.ahdploss = 0.
        self.logger.labelloss = 0.
        self.logger.stacklength = 0
        self.logger.nosibling = 0
        self.logger.totaltransition = 0
        self.initParams()


    def save(self, filename):
        self.model.save(filename)


    def load(self, filename):
        self.model.load(filename)


    def initWithExternal(self, filename):
        #  external_embedding_fp = open(filename,'r')
        count = 0
        with open(filename, "r", encoding='utf-8', errors='ignore') as f:
            for l in f:
                line = l.strip().split()
                if len(line) < 3:
                    continue
                if line[0] in self.vocab:
                    self._word_lookup.init_row(self.vocab[line[0]], [float(s) for s in line[1:]])
                    count += 1
        print(count, "hit with vocab size", len(self.vocab))

        print('Load external embedding. Vector dimensions', self.wdims)


    def initParams(self):
        self.epoch = 0
        self.bilstm = BiRNNBuilder(self.bilstm_layers, self.wdims + self.pdims, self.bildims, self.model, LSTMBuilder)
        self._word_lookup = self.model.add_lookup_parameters((len(self.vocab) + 1, self.wdims))
        if self.pdims > 0:
            self._pos_lookup = self.model.add_lookup_parameters((len(self.pos) + 1, self.pdims))

        self.graph_head_mlp = MultiLayerPerceptron([self.bildims] + [self.graph_mlpdim] * self.graph_mlplayers, self.graph_mlpactivation, self.model)
        self.graph_mod_mlp = MultiLayerPerceptron([self.bildims] + [self.graph_mlpdim] * self.graph_mlplayers, self.graph_mlpactivation, self.model)
        self.graph_bilinear = Bilinear(self.graph_mlpdim, self.model)
        self.graph_head_bias = Dense(self.graph_mlpdim, 1, identity, self.model)

        self.labeler_head_mlp = MultiLayerPerceptron([self.bildims] + [self.labeler_mlpdim] * self.labeler_mlplayers, self.labeler_mlpactivation, self.model)
        self.labeler_mod_mlp = MultiLayerPerceptron([self.bildims] + [self.labeler_mlpdim] * self.labeler_mlplayers, self.labeler_mlpactivation, self.model)
        self.labeler_concat_mlp = MultiLayerPerceptron([self.labeler_mlpdim * 2] + [self.labeler_concatdim] * self.labeler_concatlayers, self.labeler_mlpactivation, self.model)
        self.labeler_final = Dense(self.labeler_concatdim, len(self.rels), identity, self.model)

        self.ah_mlp = MultiLayerPerceptron([self.bildims * (self.stack_features+self.buffer_features)] + [self.ah_mlpdim] * self.ah_mlplayers, self.ah_mlpactivation, self.model)
        self.ah_final = Dense(self.ah_mlpdim, 3, identity, self.model)

        self.as_mlp = MultiLayerPerceptron([self.bildims * (self.stack_features+self.buffer_features)] + [self.as_mlpdim] * self.as_mlplayers, self.as_mlpactivation, self.model)
        self.as_final = Dense(self.as_mlpdim, 4, identity, self.model)

        self.ae_mlp = MultiLayerPerceptron([self.bildims * (self.stack_features+self.buffer_features)] + [self.ae_mlpdim] * self.ae_mlplayers, self.ae_mlpactivation, self.model)
        self.ae_final = Dense(self.ae_mlpdim, 4, identity, self.model)

        self.empty = [self.model.add_parameters(self.bildims) for i in range(self.stack_features+self.buffer_features)]


    def nextEpoch(self):
        self.epoch += 1
        self.trainer.update_epoch()


    def getLSTMFeatures(self, sentence, train=False):
        # sentence is the variable holding the sentence information
        # carriers is holding all vectors and needed computations
        # sentence should be left unchanged
        carriers = [ComputationCarrier() for i in range(len(sentence))]

        for entry, cc in zip(sentence, carriers):
            c = float(self.freq.get(entry.norm, 0))
            dropFlag = not train or (random.random() < (c / (0.25 + c)))
            wordvec = lookup(self._word_lookup, int(self.vocab.get(entry.norm, 0)) if dropFlag else 0) if self.wdims > 0 else None
            posvec = lookup(self._pos_lookup, int(self.pos.get(entry.pos,0))) if self.pdims > 0 else None
            cc.vec = concatenate(list(filter(None, [wordvec, posvec])))

        ret = self.bilstm.transduce([x.vec for x in carriers])
        for vec, cc in zip(ret, carriers):
            cc.vec = vec
        return carriers


    def _mstEvaluate(self, sentence):
        for i in range(len(sentence)):
            sentence[i].headmlp = self.graph_head_mlp(sentence[i].vec)
            sentence[i].modmlp = self.graph_mod_mlp(sentence[i].vec)

        exprs = [ [self.graph_bilinear(sentence[i].headmlp, sentence[j].modmlp) + \
                self.graph_head_bias(sentence[i].headmlp) for j in range(len(sentence))] for i in range(len(sentence)) ]
        scores = np.array([ [output.scalar_value() for output in exprsRow] for exprsRow in exprs ])

        for i in range(len(sentence)):
            scores[i, i] = 0.

        return scores, np.array(exprs)


    def mstSentenceLoss(self, graph, carriers):
        scores, exprs = self._mstEvaluate(carriers)
        rheads = graph.heads
        scores += 1.
        for m, h in enumerate(graph.heads):
            scores[h, m] -= 1.
        heads = parse_proj(scores)
        loss = [(exprs[h, i] - exprs[g, i]) for i, (h,g) in enumerate(zip(heads, rheads)) if h != g]

        return loss


    def mstPredict(self, graph):
        self.initCG()
        graph = graph.cleaned()

        carriers = self.getLSTMFeatures(graph.nodes)
        scores, exprs = self._mstEvaluate(carriers)
        graph.heads = parse_proj(scores)

        return graph

    def mstScores(self, graph):
        self.initCG()
        graph = graph.cleaned()

        carriers = self.getLSTMFeatures(graph.nodes)
        scores, exprs = self._mstEvaluate(carriers)

        return scores


    def _evaluateLabelPre(self, sentence):
        for i in range(len(sentence)):
            sentence[i].labeler_head_mlp = self.labeler_head_mlp(sentence[i].vec)
            sentence[i].labeler_mod_mlp = self.labeler_mod_mlp(sentence[i].vec)


    def _evaluateLabel(self, sentence, i, j):
        exprs = self.labeler_final(self.labeler_concat_mlp(concatenate([sentence[i].labeler_head_mlp, sentence[j].labeler_mod_mlp])))
        return exprs.value(), exprs


    def labelerSentenceLoss(self, graph, carriers):
        loss = []
        self._evaluateLabelPre(carriers)
        for mod, head in enumerate(graph.heads):
            if mod > 0 and head >= 0:
                scores, exprs = self._evaluateLabel(carriers, head, mod)
                goldLabelInd = self.rels[graph.rels[mod]]
                wrongLabelInd = max(((l, scr) for l, scr in enumerate(scores) if l != goldLabelInd), key=itemgetter(1))[0]
                if scores[goldLabelInd] < scores[wrongLabelInd] + 1.:
                    loss.append(exprs[wrongLabelInd] - exprs[goldLabelInd])
        return loss


    def labelPredict(self, graph):
        self.initCG()

        carriers = self.getLSTMFeatures(graph.nodes)
        self._evaluateLabelPre(carriers)
        for mod, head in enumerate(graph.heads):
            if mod > 0 and head >= 0:
                scores, exprs = self._evaluateLabel(carriers, head, mod)
                predictInd = max(enumerate(scores), key=itemgetter(1))[0]
                graph.rels[mod] = self.irels[predictInd]
        return graph


    def _ahEvaluate(self, features, sentence):
        input_vec = [sentence[f].vec if f >= 0 else parameter(self.empty[i]) for i, f in enumerate(features)]
        exprs = self.ah_final(self.ah_mlp(concatenate(input_vec)))
        if self.ah_nonpos:
            exprs = log(logistic(exprs))
        return exprs.value(), exprs

    def ahSentenceLoss(self, graph, carriers):
        loss = []
        beamconf = AHBeamConfiguration(len(graph.nodes), 1, np.array(graph.heads), self.stack_features, self.buffer_features)
        beamconf.initconf(0, self.root_first)

        while not beamconf.isComplete(0):
            costs = beamconf.transitionCosts(0)
            scores, exprs = self._ahEvaluate(beamconf.extractFeatures(0), carriers)
            best, bestCost = min(((i, c) for i, c in enumerate(costs) if c >= 0), key=itemgetter(1))
            best, bestScore = max(((i, s) for i, s in enumerate(scores) if costs[i] == bestCost), key=itemgetter(1))
            rest = tuple((i, s) for i, s in enumerate(scores) if costs[i] >= 0 and costs[i] != bestCost)
            if len(rest) == 0:
                second, secondScore = best, bestScore
            else:
                second, secondScore = max(rest, key=itemgetter(1))

            if len(rest) > 0 and scores[best] < scores[second] + 1.0:
                loss.append(exprs[second] - exprs[best])

            transition = best if ((not self.dynamic_oracle) or (len(rest) == 0) or (scores[best] - scores[second] > 1.0) or \
                    (scores[best] > scores[second] and random.random() > self.dynamic_explore_rate)) else second

            beamconf.makeTransition(0, transition)

        return loss

    def ahPredict(self, graph):
        self.initCG()
        graph = graph.cleaned()
        carriers = self.getLSTMFeatures(graph.nodes)
        beamconf = AHBeamConfiguration(len(graph.nodes), 1, np.array(graph.heads), self.stack_features, self.buffer_features)
        beamconf.initconf(0, self.root_first)
        maxstack = 0

        while not beamconf.isComplete(0):
            stacklen = beamconf.stackLength(0)
            if stacklen > maxstack:
                maxstack = stacklen
            valid = beamconf.validTransitions(0)
            scores, exprs = self._ahEvaluate(beamconf.extractFeatures(0), carriers)
            best, bestscore = max(((i, s) for i, s in enumerate(scores) if valid[i]), key=itemgetter(1))
            beamconf.makeTransition(0, best)

        graph.heads = list(beamconf.getHeads(0))
        self.logger.stacklength += maxstack

        return graph

    def _aeEvaluate(self, features, sentence):
        input_vec = [sentence[f].vec if f >= 0 else parameter(self.empty[i]) for i, f in enumerate(features)]
        exprs = self.ae_final(self.ae_mlp(concatenate(input_vec)))

        return exprs.value(), exprs

    def aeSentenceLoss(self, graph, carriers):
        loss = []
        beamconf = AEBeamConfiguration(len(graph.nodes), 1, np.array(graph.heads), self.stack_features, self.buffer_features)
        beamconf.initconf(0, self.root_first)

        while not beamconf.isComplete(0):
            valid = beamconf.validTransitions(0)
            if np.count_nonzero(valid) < 1:
                break

            costs = beamconf.transitionCosts(0)
            scores, exprs = self._aeEvaluate(beamconf.extractFeatures(0), carriers)
            best, bestCost = min(((i, c) for i, c in enumerate(costs) if c >= 0), key=itemgetter(1))
            best, bestScore = max(((i, s) for i, s in enumerate(scores) if costs[i] == bestCost), key=itemgetter(1))
            rest = tuple((i, s) for i, s in enumerate(scores) if costs[i] >= 0 and costs[i] != bestCost)
            if len(rest) == 0:
                second, secondScore = best, bestScore
            else:
                second, secondScore = max(rest, key=itemgetter(1))

            if len(rest) > 0 and scores[best] < scores[second] + 1.0:
                loss.append(exprs[second] - exprs[best])

            transition = best if ((not self.dynamic_oracle) or (len(rest) == 0) or (scores[best] - scores[second] > 1.0) or \
                    (scores[best] > scores[second] and random.random() < self.dynamic_explore_rate)) else second

            beamconf.makeTransition(0, transition)

        return loss

    def aePredict(self, graph):
        self.initCG()
        graph = graph.cleaned()
        carriers = self.getLSTMFeatures(graph.nodes)
        beamconf = AEBeamConfiguration(len(graph.nodes), 1, np.array(graph.heads), self.stack_features, self.buffer_features)
        beamconf.initconf(0, self.root_first)

        while not beamconf.isComplete(0):
            valid = beamconf.validTransitions(0)
            if np.count_nonzero(valid) < 1:
                break
            scores, exprs = self._aeEvaluate(beamconf.extractFeatures(0), carriers)
            best, bestscore = max(((i, s) for i, s in enumerate(scores) if valid[i]), key=itemgetter(1))
            beamconf.makeTransition(0, best)

        graph.heads = [i if i > 0 else 0 for i in list(beamconf.getHeads(0))]

        return graph

    def _aeBeamSeqUpdate(self, correctseq, wrongseq, beamconf, loss, carriers, loc=0, scale=1.0, direction=0):
        commonprefix = 0
        for i in range(min(len(correctseq), len(wrongseq))):
            if wrongseq[i] == correctseq[i]:
                commonprefix = i + 1
            else:
                break

        root_first = self.root_first
        beamconf.initconf(loc, root_first, direction)
        for i in range(commonprefix):
            beamconf.makeTransition(loc, wrongseq[i])
        for i in range(commonprefix, len(wrongseq)):
            feats = beamconf.extractFeatures(loc)
            scores, exprs = self._aeEvaluate(feats, carriers)
            loss.append(exprs[int(wrongseq[i])]*scale)
            beamconf.makeTransition(loc, wrongseq[i])

        beamconf.initconf(loc, root_first, direction)
        for i in range(commonprefix):
            beamconf.makeTransition(loc, correctseq[i])
        for i in range(commonprefix, len(correctseq)):
            feats = beamconf.extractFeatures(loc)
            scores, exprs = self._aeEvaluate(feats, carriers)
            loss.append(-exprs[int(correctseq[i])]*scale)
            beamconf.makeTransition(loc, correctseq[i])

    def aeDPSentenceLoss(self, graph, carriers):
        loss = []
        beamconf = AEBeamConfiguration(len(graph.nodes), 1, np.array(graph.heads), self.stack_features, self.buffer_features)
        beamconf.initconf(0, self.root_first)

        mstScores = np.ones((len(graph.nodes), len(graph.nodes)))
        for m, h in enumerate(graph.heads):
            mstScores[h, m] -= 1.

        rows = list(range(-1, len(graph.nodes))) + [-1]
        scores = np.zeros((len(rows), len(rows), 4))
        for i, ii in enumerate(rows):
            for j, jj in enumerate(rows):
                if i < j:
                    scores[i, j] = self._aeEvaluate([ii, jj], carriers)[0]

        pred_transitions, pred_heads = parse_ae_dp_mst(scores, mstScores)

        true_transitions = beamconf.goldTransitions(0, self.root_first)

        self._aeBeamSeqUpdate(true_transitions, pred_transitions, beamconf, loss, carriers, loc=0)

        return loss

    def aeDPPredict(self, graph):
        self.initCG()
        self.logger.rels = graph.rels
        carriers = self.getLSTMFeatures(graph.nodes)
        beamconf = AEBeamConfiguration(len(graph.nodes), 1, np.array(graph.heads), self.stack_features, self.buffer_features)
        beamconf.initconf(0, self.root_first)

        rows = list(range(-1, len(graph.nodes))) + [-1]
        scores = np.zeros((len(rows), len(rows), 4))
        for i, ii in enumerate(rows):
            for j, jj in enumerate(rows):
                if i < j:
                    scores[i, j] = self._aeEvaluate([ii, jj], carriers)[0]
        transitions, heads = parse_ae_dp_mst(scores, np.zeros((len(graph.nodes), len(graph.nodes))))

        for i in transitions:
            beamconf.makeTransition(0, i)

        graph = graph.cleaned()
        graph.heads = list(beamconf.getHeads(0))

        return graph

    def ahDPSentenceLoss(self, graph, carriers):
        loss = []
        beamconf = AHBeamConfiguration(len(graph.nodes), 1, np.array(graph.heads), self.stack_features, self.buffer_features)
        beamconf.initconf(0, self.root_first)

        mstScores = np.ones((len(graph.nodes), len(graph.nodes)))
        for m, h in enumerate(graph.heads):
            mstScores[h, m] -= 1.

        rows = list(range(-1, len(graph.nodes))) + [-1]
        scores = np.zeros((len(rows), len(rows), 3))
        exprs = np.zeros((len(rows), len(rows)), dtype=object)
        for i, ii in enumerate(rows):
            for j, jj in enumerate(rows):
                if (ii < 0 or jj < 0) or (ii < jj):
                    scores[i, j], exprs[i, j] = self._ahEvaluate([ii, jj], carriers)

        pred_transitions, pred_heads = parse_ah_dp_mst(scores, mstScores)

        true_transitions = beamconf.goldTransitions(0, self.root_first)

        self._ahDPSeqUpdate(true_transitions, pred_transitions, beamconf, loss, carriers, exprs, loc=0)

        return loss

    def ahDPPredict(self, graph):
        self.initCG()
        carriers = self.getLSTMFeatures(graph.nodes)

        rows = list(range(-1, len(graph.nodes))) + [-1]
        #  scores = np.array([[self._ahEvaluate([i, j], carriers)[0] for j in rows] for i in rows])
        scores = np.zeros((len(rows), len(rows), 3))
        for i, ii in enumerate(rows):
            for j, jj in enumerate(rows):
                if (ii < 0 or jj < 0) or (ii < jj):
                    scores[i, j] = self._ahEvaluate([ii, jj], carriers)[0]

        transitions, heads = parse_ah_dp_mst(scores, np.zeros((len(graph.nodes), len(graph.nodes))))

        graph = graph.cleaned()
        graph.heads = heads

        return graph

    def ahDPScores(self, graph):
        self.initCG()
        carriers = self.getLSTMFeatures(graph.nodes)
        rows = list(range(-1, len(graph.nodes))) + [-1]
        scores = np.array([[self._ahEvaluate([i, j], carriers)[0] for j in rows] for i in rows])

        return scores

        return graph


    def _ahDPSeqUpdate(self, correctseq, wrongseq, beamconf, loss, carriers, exprs, loc=0):
        commonprefix = 0
        for i in range(min(len(correctseq), len(wrongseq))):
            if wrongseq[i] == correctseq[i]:
                commonprefix = i + 1
            else:
                break

        beamconf.initconf(loc, self.root_first)
        for i in range(commonprefix):
            beamconf.makeTransition(loc, wrongseq[i])

        for i in range(commonprefix, len(wrongseq)):
            ii, jj = beamconf.extractFeatures(loc)
            loss.append(exprs[ii + 1, jj + 1][int(wrongseq[i])])
            beamconf.makeTransition(loc, wrongseq[i])

        beamconf.initconf(loc, self.root_first)
        for i in range(commonprefix):
            beamconf.makeTransition(loc, correctseq[i])
        for i in range(commonprefix, len(correctseq)):
            ii, jj = beamconf.extractFeatures(loc)
            loss.append(-exprs[ii + 1, jj + 1][int(correctseq[i])])
            beamconf.makeTransition(loc, correctseq[i])


    def _minibatchUpdate(self, loss, num_tokens):
        if len(loss) == 0:
            self.initCG(train=True)
            return 0.
        eerrs = esum(loss) * (1./ num_tokens)
        ret = eerrs.scalar_value()
        eerrs.backward()
        self.trainer.update()
        self.initCG(train=True)
        return ret * num_tokens


    def train(self, graphs, **kwargs):
        print(kwargs)
        mst = kwargs.get("mst", False)
        ah = kwargs.get("ah", False)
        ahDP = kwargs.get("ahDP", False)
        ae = kwargs.get("ae", False)
        aeDP = kwargs.get("aeDP", False)
        label = kwargs.get("label", False)
        self.trainer.status()
        self.initCG(train=True)

        loss = []
        loss_sum = 0.0
        num_tokens = 0
        t0 = time.time()

        for i, (j, graph) in enumerate(random.sample(list(enumerate(graphs)), len(graphs))):
            if self.verbose:
                print(i, end=" ")
                sys.stdout.flush()
            elif (i+1) % 100 == 0:
                print(i + 1, "{0:.2f}s".format(time.time() - t0), end=" ")
                sys.stdout.flush()
                t0 = time.time()
            isProjective = graph.isProjective()
            if len(graph.nodes) <= 2 or (not isProjective):
                continue
            carriers = self.getLSTMFeatures(graph.nodes, train=True)
            num_tokens += len(graph.nodes) - 1
            if mst:
                loss.extend(self.mstSentenceLoss(graph, carriers))

            if ah:
                loss.extend(self.ahSentenceLoss(graph, carriers))

            if ahDP:
                loss.extend(self.ahDPSentenceLoss(graph, carriers))

            if ae:
                loss.extend(self.aeSentenceLoss(graph, carriers))

            if aeDP:
                loss.extend(self.aeDPSentenceLoss(graph, carriers))

            if label:
                loss.extend(self.labelerSentenceLoss(graph, carriers))

            if num_tokens >= self.batch_size:
                loss_sum += self._minibatchUpdate(loss, num_tokens)
                loss = []
                num_tokens = 0

        loss_sum += self._minibatchUpdate(loss, num_tokens)
        self.nextEpoch()
        print()
        print("Total Loss", loss_sum, "Avg", loss_sum / len(graphs))

    def initCG(self, train=False):
        renew_cg()
        if train:
            self.graph_head_mlp.set_dropout(self.graph_mlpdropout)
            self.graph_mod_mlp.set_dropout(self.graph_mlpdropout)
            self.labeler_head_mlp.set_dropout(self.labeler_mlpdropout)
            self.labeler_mod_mlp.set_dropout(self.labeler_mlpdropout)
            self.labeler_concat_mlp.set_dropout(self.labeler_concatdropout)
            self.ah_mlp.set_dropout(self.ah_mlpdropout)
            self.ae_mlp.set_dropout(self.ae_mlpdropout)
            self.bilstm.set_dropout(self.bilstm_dropout)
        else:
            self.graph_head_mlp.set_dropout(0.)
            self.graph_mod_mlp.set_dropout(0.)
            self.labeler_head_mlp.set_dropout(0.)
            self.labeler_mod_mlp.set_dropout(0.)
            self.labeler_concat_mlp.set_dropout(0.)
            self.ah_mlp.set_dropout(0.)
            self.ae_mlp.set_dropout(0.)
            self.bilstm.set_dropout(0.)
