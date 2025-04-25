from openjij import SASampler, SQASampler
from collections import defaultdict, Counter
import numpy as np

class MagicCircle:
    def __init__(self, N=3):
        self.N = N     # self.N = 3
        self.M = N * N     # self.M = 9
        self.S = N * (N**2 + 1) // 2   # self.S = 15
        self.idx = {}
        k = 0
        for i in range(self.N):
            for j in range(self.N):
                for n in range(self.M):
                    self.idx[(i,j,n)] = k
                    k += 1
        samplers = [SASampler(), SQASampler()]
        self.sampler = samplers[0]

    def get_param(self):
        return self.N, self.M, self.S, self.idx

    def sub1(self, i, j, L, Q):
        N, M, _, idx = self.get_param()
        for n1 in range(M):
            Q[(idx[(i, j, n1)], idx[(i, j, n1)])] -= 2.0 * L
            for n2 in range(M):
                Q[(idx[(i, j, n1)], idx[(i, j, n2)])] += 1.0 * L

    def f1(self, L, Q):
        N, _, _, _ = self.get_param()
        for i in range(N):
            for j in range(N):
                self.sub1(i, j, L, Q)
        return Q

    def sub2(self, n, L, Q):
        N, _, _, idx = self.get_param()
        for i1 in range(N):
            for j1 in range(N):
                Q[(idx[(i1, j1, n)], idx[(i1, j1, n)])] -= 2.0 * L
                for i2 in range(N):
                    for j2 in range(N):
                        Q[(idx[(i1, j1, n)], idx[(i2, j2, n)])] += 1.0 * L

    def f2(self, L, Q):
        _, M, _, _ = self.get_param()
        for n in range(M):
            self.sub2(n, L, Q)
        return Q

    def sub3(self, i, L, Q):
        N, M, S, idx = self.get_param()
        for j1 in range(N):
            for n1 in range(M):
                Q[(idx[(i, j1 ,n1)], idx[(i, j1, n1)])] -= 2.0 * (n1+1) * S * L
                for j2 in range(N):
                    for n2 in range(M):
                        Q[(idx[(i, j1, n1)], idx[(i, j2, n2)])] += (n1+1) * (n2+1) * L

    def f3(self, L, Q):
        N, _, _, _ = self.get_param()
        for i in range(N):
            self.sub3(i, L, Q)
        return Q

    def sub4(self, j, L, Q):
        N, M, S, idx = self.get_param()
        for i1 in range(N):
            for n1 in range(M):
                Q[(idx[(i1, j, n1)], idx[(i1, j, n1)])] -= 2.0 * (n1+1) * S * L
                for i2 in range(N):
                    for n2 in range(M):
                        Q[(idx[(i1, j, n1)], idx[(i2, j, n2)])] += (n1+1) * (n2+1) * L

    def f4(self, L, Q):
        N, _, _, _ = self.get_param()
        Q = defaultdict(lambda: 0)
        for j in range(N):
            self.sub4(j, L, Q)
        return Q

    def f5(self, L, Q):
        N, M, S, idx = self.get_param()
        Q = defaultdict(lambda: 0)
        for d1 in range(N):
            for n1 in range(M):
                Q[(idx[(d1, d1, n1)], idx[(d1, d1, n1)])] -= 2.0 * (n1+1) * S * L
                Q[(idx[(d1, N-d1-1, n1)], idx[(d1, N-d1-1, n1)])] -= 2.0 * (n1 + 1) * S * L
                for d2 in range(N):
                    for n2 in range(M):
                        Q[(idx[(d1, d1, n1)], idx[(d2, d2, n2)])] += (n1+1) * (n2+1) * L
                        Q[(idx[(d1, N-d1-1, n1)], idx[(d2, N-d2-1, n2)])] += (n1 + 1) * (n2 + 1) * L
        return Q

    def f(self, lagrange1=1.0, lagrange2=1.0, lagrange3=1.0):
        Q = defaultdict(lambda: 0)
        _ = self.f1(lagrange1, Q)
        _ = self.f2(lagrange2, Q)
        _ = self.f3(lagrange3, Q)
        _ = self.f4(lagrange3, Q)
        _ = self.f5(lagrange3, Q)
        return Q

    def solv(self, Q, num_reads=1):
        sampleset = self.sampler.sample_qubo(Q, num_reads=num_reads)
        return sampleset

    def result(self, sampleset):
        N, M, S, idx = self.get_param()
        result = [i for i in sampleset.first[0].values()]
        ans = [[None] * N for _ in range(N)]
        for i in range(N):
            for j in range(N):
                for n in range(N**2):
                    if result[idx[(i,j,n)]] == 1:
                        ans[i][j] = n+1
        return ans

if __name__ == '__main__':
    mc = MagicCircle(3)
    Q = mc.f(10.0, 10.0, 1.0)
    num_reads = 1000
    sampleset = mc.solv(Q, num_reads)
    ans = mc.result(sampleset)
    print(*ans, sep='\n')
