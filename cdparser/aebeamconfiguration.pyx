#!/usr/bin/env python
# encoding: utf-8

cimport cython
cimport numpy as np
import numpy as np
from libc.stdlib cimport malloc, free

np.import_array()

cdef np.float64_t NEGINF = -np.inf
cdef np.float64_t INF = np.inf
cdef inline int int_max(int a, int b): return a if a >= b else b
cdef inline int int_min(int a, int b): return a if a <= b else b
cdef inline np.float64_t float64_max(np.float64_t a, np.float64_t b): return a if a >= b else b
cdef inline np.float64_t float64_min(np.float64_t a, np.float64_t b): return a if a <= b else b

cdef int STACK_LOC = 0, BUFFER_LOC = 1, RESOLVED = 2
cdef int SHIFT = 0, LEFTARC = 1, RIGHTARC = 2, REDUCE = 3
cdef int L2R = 0, R2L = 1


@cython.boundscheck(False)
@cython.wraparound(False)
cdef class AEBeamConfiguration:

    cdef int size
    cpdef int length
    cdef int *step
    cdef int *transitions
    cdef int *heads
    cdef int *stack
    cdef int *buffer
    cdef int *stack_head
    cdef int *buffer_head
    cdef int *location
    cdef float *transition_score
    cdef int *cost
    cdef int *gold_heads
    cdef int *root_first
    cdef int *direction
    cdef int stack_features
    cdef int buffer_features

    def __cinit__(self, int length, int size, np.ndarray[ndim=1, dtype=np.npy_intp] gold_heads, int stack_features, int buffer_features):
        self.size = size
        self.length = length
        self.stack_features = stack_features
        self.buffer_features = buffer_features
        self.step = <int *>malloc(size*sizeof(int))
        self.transitions = <int *>malloc(size*length*2*sizeof(int))
        self.heads = <int *>malloc(size*length*sizeof(int))
        self.stack = <int *>malloc(size*length*sizeof(int))
        self.buffer = <int *>malloc(size*length*sizeof(int))
        self.stack_head = <int *>malloc(size*sizeof(int))
        self.buffer_head = <int *>malloc(size*sizeof(int))
        self.location = <int *>malloc(size*length*sizeof(int))
        self.transition_score = <float *>malloc(size*sizeof(float))
        self.cost = <int *>malloc(size*sizeof(int))
        self.gold_heads = <int *>malloc(length*sizeof(int))
        self.root_first = <int *>malloc(size*sizeof(int))
        self.direction = <int *>malloc(size*sizeof(int))

        cdef int i
        for i in range(length):
            self.gold_heads[i] = gold_heads[i]

    def __dealloc__(self):
        free(self.step)
        free(self.transitions)
        free(self.heads)
        free(self.stack)
        free(self.buffer)
        free(self.stack_head)
        free(self.buffer_head)
        free(self.location)
        free(self.transition_score)
        free(self.cost)
        free(self.gold_heads)
        free(self.root_first)
        free(self.direction)

    cpdef void copy(self, int source, int target):
        cdef int i
        cdef int length = self.length
        self.step[target] = self.step[source]
        self.root_first[target] = self.root_first[source]
        self.direction[target] = self.direction[source]
        self.stack_head[target] = self.stack_head[source]
        self.buffer_head[target] = self.buffer_head[source]
        self.transition_score[target] = self.transition_score[source]
        self.cost[target] = self.cost[source]
        for i in range(self.length):
            self.heads[target*length+i] = self.heads[source*length+i]
            self.stack[target*length+i] = self.stack[source*length+i]
            self.buffer[target*length+i] = self.buffer[source*length+i]
            self.location[target*length+i] = self.location[source*length+i]
        for i in range(self.length*2):
            self.transitions[target*length*2+i] = self.transitions[source*length*2+i]

    cpdef void initconf(self, int pos, int rootfirst=0, int direction=L2R):
        cdef int i
        cdef int length = self.length
        self.step[pos] = 0
        self.stack_head[pos] = 0
        self.buffer_head[pos] = length
        self.transition_score[pos] = 0.0
        self.cost[pos] = 0
        self.root_first[pos] = rootfirst
        self.direction[pos] = direction
        for i in range(length):
            self.heads[pos*length+i] = -1
            self.location[pos*length+i] = BUFFER_LOC

        if direction == L2R:
            for i in range(length):
                if rootfirst:
                    self.buffer[pos*length+i] = length - i - 1
                else:
                    self.buffer[pos*length+i] = length - i
            if not rootfirst:
                self.buffer[pos*length] = 0
        else:
            for i in range(length):
                if rootfirst:
                    self.buffer[pos*length+i] = i
                else:
                    self.buffer[pos*length+i] = i + 1
            if not rootfirst:
                self.buffer[pos*length + length - 1] = 0


    cpdef np.ndarray[ndim=1, dtype=np.npy_intp] goldTransitions(self, int pos, int rootfirst=0, int direction=L2R):
        cdef np.ndarray[ndim=1, dtype=np.npy_intp] transitions
        cdef np.ndarray[ndim=1, dtype=np.npy_intp] costs
        cdef int i, j

        transitions = np.full(2 * self.length - 1, 0, dtype=int)

        self.initconf(pos, rootfirst, direction)
        for i in range(0, 2 * self.length - 1):
            costs = self.transitionCosts(pos)
            for j in range(0, 4):
                if costs[j] == 0:
                    transitions[i] = j
                    self.makeTransition(pos, j)
                    break
        return transitions

    cpdef int stackLength(self, int pos):
        return self.stack_head[pos]

    def fullInfo(self, int pos):
        cdef int i
        cdef int offset = self.length * pos
        return (self.step[pos], self.transition_score[pos], self.cost[pos],
                [self.gold_heads[i] for i in range(self.length)],
                [self.heads[offset+i] for i in range(self.length)],
                [self.stack[offset+i] for i in range(self.stack_head[pos])],
                [self.buffer[offset+i] for i in range(self.buffer_head[pos])])

    def getTransitions(self, int pos):
        cdef int offset = self.length * 2 * pos
        cdef int i
        cdef int step = self.step[pos]
        cdef np.ndarray[ndim=1, dtype=np.npy_intp] ret
        ret = np.empty(step, dtype=int)
        for i in range(step):
            ret[i] = self.transitions[offset + i]
        return ret

    def getHeads(self, int pos):
        cdef int offset = self.length * pos
        cdef int i
        cdef np.ndarray[ndim=1, dtype=np.npy_intp] ret
        ret = np.empty(self.length, dtype=int)
        for i in range(self.length):
            ret[i] = self.heads[offset + i]
        return ret

    cpdef float getScore(self, int pos):
        return self.transition_score[pos]

    cpdef void transitionScoreDev(self, int pos, float dev):
        self.transition_score[pos] += dev

    def extractFeatures(self, int pos):
        cdef np.ndarray[ndim=1, dtype=np.npy_intp] ret
        cdef int total, i
        cdef int offset = self.length * pos
        cdef int nostack = self.stack_features, nobuffer = self.buffer_features

        total = nostack + nobuffer

        ret = np.full(total, -1, dtype=int)
        for i in range(nostack):
            if self.stack_head[pos] > i:
                ret[i] = self.stack[offset+self.stack_head[pos]-i-1]
        for i in range(nobuffer):
            if self.buffer_head[pos] > i:
                ret[nostack+i] = self.buffer[offset+self.buffer_head[pos]-i-1]

        return ret

    def isComplete(self, int pos):
        if self.buffer_head[pos] == 0 and self.stack_head[pos] == 1:
            return True
        else:
            return False

    cpdef np.ndarray[ndim=1, dtype=np.npy_intp] validTransitions(self, int pos):
        cdef int offset = self.length * pos
        cdef np.ndarray[ndim=1, dtype=np.npy_intp] ret
        ret = np.zeros(4, dtype=int)
        if self.buffer_head[pos] > 0 and (self.stack_head[pos] == 0 or self.buffer[offset+self.buffer_head[pos]-1] != 0):
            ret[SHIFT] = 1
        if self.buffer_head[pos] > 0 and self.stack_head[pos] > 0 and \
                self.buffer[offset+self.buffer_head[pos]-1] != 0:
            ret[RIGHTARC] = 1
        if self.stack_head[pos] > 0 and self.buffer_head[pos] > 0 and \
                self.stack[offset+self.stack_head[pos]-1] != 0 and \
                self.heads[offset+self.stack[offset+self.stack_head[pos]-1]] < 0:
            ret[LEFTARC] = 1
        if self.stack_head[pos] > 0 and self.heads[offset+self.stack[offset+self.stack_head[pos]-1]] >= 0:
            ret[REDUCE] = 1
        return ret

    cpdef void makeTransition(self, int pos, int transition):
        if transition == SHIFT:
            self.shift(pos)
        elif transition == LEFTARC:
            self.leftarc(pos)
        elif transition == RIGHTARC:
            self.rightarc(pos)
        elif transition == REDUCE:
            self.reduce(pos)

    cpdef void shift(self, int pos):
        cdef int offset = self.length * pos
        cdef int b0 = self.buffer[offset+self.buffer_head[pos]-1]
        self.location[offset+b0] = STACK_LOC
        self.stack[offset+self.stack_head[pos]] = b0
        self.buffer_head[pos] -= 1
        self.stack_head[pos] += 1
        self.transitions[offset*2+self.step[pos]] = SHIFT
        self.step[pos] += 1

    cpdef void leftarc(self, int pos):
        cdef int offset = self.length * pos
        cdef int s0 = self.stack[offset+self.stack_head[pos]-1]
        cdef int b0 = self.buffer[offset+self.buffer_head[pos]-1]
        self.location[offset+s0] = RESOLVED
        self.heads[offset+s0] = b0
        self.stack_head[pos] -= 1
        self.transitions[offset*2+self.step[pos]] = LEFTARC
        self.step[pos] += 1

    cpdef void rightarc(self, int pos):
        cdef int offset = self.length * pos
        cdef int s0 = self.stack[offset+self.stack_head[pos]-1]
        cdef int b0 = self.buffer[offset+self.buffer_head[pos]-1]
        self.location[offset+b0] = STACK_LOC
        self.heads[offset+b0] = s0
        self.stack[offset+self.stack_head[pos]] = b0
        self.stack_head[pos] += 1
        self.buffer_head[pos] -= 1
        self.transitions[offset*2+self.step[pos]] = RIGHTARC
        self.step[pos] += 1

    cpdef void reduce(self, int pos):
        cdef int offset = self.length * pos
        cdef int s0 = self.stack[offset+self.stack_head[pos]-1]
        self.location[offset+s0] = RESOLVED
        self.stack_head[pos] -= 1
        self.transitions[offset*2+self.step[pos]] = REDUCE
        self.step[pos] += 1

    cpdef np.ndarray[ndim=1, dtype=np.npy_intp] transitionCosts(self, int pos):
        cdef int offset = self.length * pos
        cdef np.ndarray[ndim=1, dtype=np.npy_intp] ret
        cdef np.ndarray[ndim=1, dtype=np.npy_intp] valid = self.validTransitions(pos)
        cdef int cost, i, parent
        cdef int s0=-1, b0=-1
        if self.stack_head[pos] > 0:
            s0 = self.stack[offset+self.stack_head[pos]-1]
        if self.buffer_head[pos] > 0:
            b0 = self.buffer[offset+self.buffer_head[pos]-1]
        ret = np.full(4, -1, dtype=int)

        if valid[SHIFT]:
            cost = 0
            if self.gold_heads[b0] >= 0 and self.location[offset+self.gold_heads[b0]] == STACK_LOC:
                cost += 1
            for i in range(self.stack_head[pos]):
                parent = self.gold_heads[self.stack[offset+i]]
                if self.heads[offset+self.stack[offset+i]] < 0 and parent >= 0 and parent == b0:
                    cost += 1
            ret[SHIFT] = cost

        if valid[LEFTARC]:
            cost = 0
            for i in range(self.buffer_head[pos]):
                parent = self.gold_heads[self.buffer[offset+i]]
                if parent >= 0 and parent == s0:
                    cost += 1
            parent = self.gold_heads[s0]
            if parent >= 0 and self.location[offset+parent] == BUFFER_LOC and parent != b0:
                cost += 1
            if self.gold_heads[s0] == 0 and self.stack_head[pos] > 0 and self.stack[0] == 0:
                cost += 1
            ret[LEFTARC] = cost

        if valid[RIGHTARC]:
            cost = 0
            for i in range(self.stack_head[pos]):
                parent = self.gold_heads[self.stack[offset+i]]
                if self.heads[offset+self.stack[offset+i]] < 0 and parent >= 0 and parent == b0:
                    cost += 1
            parent = self.gold_heads[b0]
            if parent >= 0 and (self.location[offset+parent] == BUFFER_LOC or \
                    (self.location[offset+parent] == STACK_LOC and parent != s0)):
                cost += 1
            ret[RIGHTARC] = cost

        if valid[REDUCE]:
            cost = 0
            for i in range(self.buffer_head[pos]):
                parent = self.gold_heads[self.buffer[offset+i]]
                if parent >= 0 and parent == s0:
                    cost += 1
            ret[REDUCE] = cost

        return ret
