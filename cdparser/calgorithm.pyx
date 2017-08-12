#!/usr/bin/env python
# encoding: utf-8

cimport cython
cimport numpy as np
import numpy as np

np.import_array()

cdef np.float64_t NEGINF = -np.inf
cdef np.float64_t INF = np.inf
cdef inline int int_max(int a, int b): return a if a >= b else b
cdef inline int int_min(int a, int b): return a if a <= b else b
cdef inline np.float64_t float64_max(np.float64_t a, np.float64_t b): return a if a >= b else b
cdef inline np.float64_t float64_min(np.float64_t a, np.float64_t b): return a if a <= b else b

@cython.boundscheck(False)
@cython.wraparound(False)
def cFindViolation(np.ndarray[ndim=1, dtype=np.float64_t] pri, np.ndarray[ndim=1, dtype=np.uint8_t] isGold):
    cdef int n_len, i
    cdef np.float64_t bestGold, bestNotGold
    cdef int goldIdx, notGoldIdx
    n_len = pri.shape[0]
    bestGold, bestNotGold = INF, INF
    goldIdx = -1
    notGoldIdx = -1

    for i in range(n_len):
        if isGold[i]:
            if bestGold > pri[i]:
                goldIdx = i
                bestGold = pri[i]
        else:
            if bestNotGold > pri[i]:
                notGoldIdx = i
                bestNotGold = pri[i]

    return goldIdx, notGoldIdx



@cython.boundscheck(False)
@cython.wraparound(False)
def cIsProjective(np.ndarray[ndim=1, dtype=np.npy_intp] heads):
    cdef int n_len, i, j, cur
    cdef int edge1_0, edge1_1, edge2_0, edge2_1
    n_len = heads.shape[0]
    for i in range(n_len):
        if heads[i] < 0:
            continue
        for j in range(i + 1, n_len):
            if heads[j] < 0:
                continue
            edge1_0 = int_min(i, heads[i])
            edge1_1 = int_max(i, heads[i])
            edge2_0 = int_min(j, heads[j])
            edge2_1 = int_max(j, heads[j])
            if edge1_0 == edge2_0:
                if edge1_1 == edge2_1:
                    return False
                else:
                    continue
            if edge1_0 < edge2_0 and not (edge2_0 >= edge1_1 or edge2_1 <= edge1_1):
                return False
            if edge1_0 > edge2_0 and not (edge1_0 >= edge2_1 or edge1_1 <= edge2_1):
                return False

    return True


@cython.boundscheck(False)
@cython.wraparound(False)
def cInsideScore(np.ndarray[ndim=1, dtype=np.npy_intp] heads, np.ndarray[ndim=2, dtype=np.float64_t] scores):
    cdef int n_len, m, h
    cdef np.float64_t f
    f = 0.
    n_len = heads.shape[0]
    for m in range(n_len):
        h = heads[m]
        if h >= 0:
            f += scores[h, m]
    return f


@cython.boundscheck(False)
@cython.wraparound(False)
def get_estimate_approx(np.ndarray[ndim=1, dtype=np.npy_intp] stack, np.ndarray[ndim=1, dtype=np.npy_intp] buffer, np.ndarray[ndim=2, dtype=np.float64_t] scores):
    cdef int i, j, m, h, n_stack, n_buffer, n_len
    cdef np.float64_t g, curmax

    g = 0.
    n_stack = stack.shape[0]
    n_buffer = buffer.shape[0]
    n_len = scores.shape[0]

    for i in range(n_buffer):
        m = buffer[i]
        if m == 0:
            continue
        curmax = NEGINF
        for j in range(n_stack):
            h = stack[j]
            curmax = float64_max(curmax, scores[h, m])
        for j in range(n_buffer):
            h = buffer[j]
            if h != m:
                curmax = float64_max(curmax, scores[h, m])
        g += curmax

    for i in range(n_stack):
        m = stack[n_stack - 1 - i]
        if m == 0:
            continue
        curmax = NEGINF
        for j in range(n_buffer):
            h = buffer[j]
            curmax = float64_max(curmax, scores[h, m])
        if i < n_stack - 1:
            h = stack[n_stack - 2 - i]
            curmax = float64_max(curmax, scores[h, m])
        g += curmax

    return g


@cython.boundscheck(False)
@cython.wraparound(False)
def get_estimate_acc(np.ndarray[ndim=1, dtype=np.npy_intp] stack, np.ndarray[ndim=1, dtype=np.npy_intp] buffer, np.ndarray[ndim=2, dtype=np.float64_t] scores):
    cdef np.ndarray[ndim=2, dtype=np.float64_t] tmpscores
    cdef np.ndarray[ndim=1, dtype=np.npy_intp] dic
    cdef int i, j, m, h, n_stack, n_buffer, n_len, n_newlen

    n_stack = stack.shape[0]
    n_buffer = buffer.shape[0]
    n_newlen = n_stack + n_buffer
    tmpscores = np.empty((n_stack + n_buffer, n_stack + n_buffer))
    for i in range(n_newlen):
        for j in range(n_newlen):
            tmpscores[i, j] = NEGINF

    n_len = scores.shape[0]
    dic = np.empty(n_len, dtype=int)
    for i in range(n_stack):
        dic[stack[i]] = i + 1
    for i in range(n_buffer):
        dic[buffer[i]] = n_newlen - i

    dic[0] = 0

    for i in range(n_buffer):
        m = buffer[i]
        if m == 0:
            continue
        for j in range(n_stack):
            h = stack[j]
            tmpscores[dic[h], dic[m]] = scores[h, m]
        for j in range(n_buffer):
            h = buffer[j]
            if h != m:
                tmpscores[dic[h], dic[m]] = scores[h, m]
    for i in range(n_stack):
        m = stack[n_stack - 1 - i]
        if m == 0:
            continue
        for j in range(n_buffer):
            h = buffer[j]
            tmpscores[dic[h], dic[m]] = scores[h, m]
        if i < n_stack - 1:
            h = stack[n_stack - 2 - i]
            tmpscores[dic[h], dic[m]] = scores[h, m]
    return parse_proj_vals(tmpscores)


@cython.boundscheck(False)
@cython.wraparound(False)
def parse_proj_vals(np.ndarray[ndim=2, dtype=np.float64_t] scores):
    cdef int nr, nc, N, i, k, s, t, r
    cdef np.float64_t tmp, cand
    cdef np.ndarray[ndim=2, dtype=np.float64_t] complete_0
    cdef np.ndarray[ndim=2, dtype=np.float64_t] complete_1
    cdef np.ndarray[ndim=2, dtype=np.float64_t] incomplete_0
    cdef np.ndarray[ndim=2, dtype=np.float64_t] incomplete_1

    nr, nc = np.shape(scores)

    N = nr - 1 # Number of words (excluding root).

    # Initialize CKY table.

    complete_0 = np.zeros((nr, nr)) # s, t, direction (right=1).
    complete_1 = np.zeros((nr, nr)) # s, t, direction (right=1).
    incomplete_0 = np.zeros((nr, nr)) # s, t, direction (right=1).
    incomplete_1 = np.zeros((nr, nr)) # s, t, direction (right=1).

    for i in range(nr):
        incomplete_0[i, 0] = NEGINF

    for k in range(1, nr):
        for s in range(nr - k):
            t = s + k
            tmp = NEGINF
            for r in range(s, t):
                cand = complete_1[s, r] + complete_0[r+1, t]
                if cand > tmp:
                    tmp = cand
            incomplete_0[t, s] = tmp + scores[t, s]
            incomplete_1[s, t] = tmp + scores[s, t]

            tmp = NEGINF
            for r in range(s, t):
                cand = complete_0[s, r] + incomplete_0[t, r]
                if cand > tmp:
                    tmp = cand
            complete_0[s, t] = tmp

            tmp = NEGINF
            for r in range(s+1, t+1):
                cand = incomplete_1[s, r] + complete_1[r, t]
                if cand > tmp:
                    tmp = cand
            complete_1[s, t] = tmp

    return complete_1[0, N]


@cython.boundscheck(False)
@cython.wraparound(False)
def parse_proj(np.ndarray[ndim=2, dtype=np.float64_t] scores):
    cdef int nr, nc, N, i, k, s, t, r, maxidx
    cdef np.float64_t tmp, cand
    cdef np.ndarray[ndim=2, dtype=np.float64_t] complete_0
    cdef np.ndarray[ndim=2, dtype=np.float64_t] complete_1
    cdef np.ndarray[ndim=2, dtype=np.float64_t] incomplete_0
    cdef np.ndarray[ndim=2, dtype=np.float64_t] incomplete_1
    cdef np.ndarray[ndim=3, dtype=np.npy_intp] complete_backtrack
    cdef np.ndarray[ndim=3, dtype=np.npy_intp] incomplete_backtrack
    cdef np.ndarray[ndim=1, dtype=np.npy_intp] heads

    nr, nc = np.shape(scores)

    N = nr - 1 # Number of words (excluding root).

    complete_0 = np.zeros((nr, nr)) # s, t, direction (right=1).
    complete_1 = np.zeros((nr, nr)) # s, t, direction (right=1).
    incomplete_0 = np.zeros((nr, nr)) # s, t, direction (right=1).
    incomplete_1 = np.zeros((nr, nr)) # s, t, direction (right=1).

    complete_backtrack = -np.ones((nr, nr, 2), dtype=int) # s, t, direction (right=1).
    incomplete_backtrack = -np.ones((nr, nr, 2), dtype=int) # s, t, direction (right=1).

    for i in range(nr):
        incomplete_0[i, 0] = NEGINF

    for k in range(1, nr):
        for s in range(nr - k):
            t = s + k
            tmp = NEGINF
            maxidx = s
            for r in range(s, t):
                cand = complete_1[s, r] + complete_0[r+1, t]
                if cand > tmp:
                    tmp = cand
                    maxidx = r
            incomplete_0[t, s] = tmp + scores[t, s]
            incomplete_1[s, t] = tmp + scores[s, t]
            incomplete_backtrack[s, t, 0] = maxidx
            incomplete_backtrack[s, t, 1] = maxidx

            tmp = NEGINF
            maxidx = s
            for r in range(s, t):
                cand = complete_0[s, r] + incomplete_0[t, r]
                if cand > tmp:
                    tmp = cand
                    maxidx = r
            complete_0[s, t] = tmp
            complete_backtrack[s, t, 0] = maxidx

            tmp = NEGINF
            maxidx = s + 1
            for r in range(s+1, t+1):
                cand = incomplete_1[s, r] + complete_1[r, t]
                if cand > tmp:
                    tmp = cand
                    maxidx = r
            complete_1[s, t] = tmp
            complete_backtrack[s, t, 1] = maxidx

    # return complete_1[0, N]
    heads = -np.ones(N + 1, dtype=int)
    backtrack_eisner(incomplete_backtrack, complete_backtrack, 0, N, 1, 1, heads)

    return heads


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void backtrack_eisner(np.ndarray[ndim=3, dtype=np.npy_intp] incomplete_backtrack, np.ndarray[ndim=3, dtype=np.npy_intp]complete_backtrack, int s, int t, int direction, int complete, np.ndarray[ndim=1, dtype=np.npy_intp] heads):
    cdef int r
    if s == t:
        return
    if complete:
        r = complete_backtrack[s, t, direction]
        if direction:
            backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 1, 0, heads)
            backtrack_eisner(incomplete_backtrack, complete_backtrack, r, t, 1, 1, heads)
            return
        else:
            backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 0, 1, heads)
            backtrack_eisner(incomplete_backtrack, complete_backtrack, r, t, 0, 0, heads)
            return
    else:
        r = incomplete_backtrack[s, t, direction]
        if direction:
            heads[t] = s
            backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 1, 1, heads)
            backtrack_eisner(incomplete_backtrack, complete_backtrack, r+1, t, 0, 1, heads)
            return
        else:
            heads[s] = t
            backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 1, 1, heads)
            backtrack_eisner(incomplete_backtrack, complete_backtrack, r+1, t, 0, 1, heads)
            return
