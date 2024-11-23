"""
Solves the assignment problem in polynomial time.
There are W workers and J tasks, each associated with a cost for assignment.
The algorithm finds the set of assignments that minimize the cumulative cost.
This algorithm doesn't assume that the jobs have a priority.

Input:
- Eigen matrix C: a JxW matrix where C(i, j) represents the cost of assigning task i to worker j.

Assumptions:
- Every task must be assigned to exactly one worker, hence J <= W (more workers than tasks).

Output:
- A vector<int> where the element at index k is the task assigned to worker k.
  For example, r[1] = 3 indicates worker 1 is assigned to task 3.

Complexity:
- The algorithm runs in O(J^2 * W).

Notes: translated from C++ from the current code for SPQR Team.
If we use that module in RoboCup 2025, it'll be released by October that year.
"""

import sys
import numpy as np

def hungarian(cost_matrix: np.ndarray) -> list:
    J, W = cost_matrix.shape
    assert J <= W, "transpose pls"

    job = [-1] * (W + 1)  # job[w] = job assigned to the w-th worker
    ys = np.zeros(J)  # Job potentials
    yt = np.zeros(W + 1)  # Worker potentials
    assignments = [-1] * W  # To store assignments

    inf = np.inf

    for j_cur in range(J):
        w_cur = W
        job[w_cur] = j_cur

        min_to = np.full(W, inf)
        prv = [-1] * W  # Previous worker on alternating path
        in_Z = [False] * (W + 1)  # Whether worker is in Z

        while job[w_cur] != -1:  # Runs at most j_cur + 1 times
            in_Z[w_cur] = True
            j = job[w_cur]
            delta = inf
            w_next = -1

            for w in range(W):
                if not in_Z[w]:
                    reduced_cost = cost_matrix[j, w] - ys[j] - yt[w]
                    if reduced_cost < min_to[w]:
                        min_to[w] = reduced_cost
                        prv[w] = w_cur
                    if min_to[w] < delta:
                        delta = min_to[w]
                        w_next = w

            for w in range(W):
                if in_Z[w]:
                    ys[job[w]] += delta  # Update potentials for assigned jobs
                    yt[w] -= delta  # Update potential for current worker
                else:
                    min_to[w] -= delta  # Update minimum reduced cost
            w_cur = w_next  # Move to the next worker

        # Update assignments along alternating path
        while w_cur != W:
            w = prv[w_cur]
            job[w_cur] = job[w]
            w_cur = w

    for w in range(W):
        assignments[w] = job[w]

    return assignments
