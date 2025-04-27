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
        for j in range(N):
            self.sub4(j, L, Q)
        return Q

    def f5(self, L, Q):
        N, M, S, idx = self.get_param()
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

    def evaluate(self, sampleset, prn=True):
        # Extract sample solutions, energies, and sort them by frequency
        samples = sampleset.record['sample']
        energies = sampleset.record['energy']
        # Combine solutions and corresponding energies
        sample_data = [(tuple(sample), energy) for sample, energy in zip(samples, energies)]
        # Sort the results by appearance frequency and then energy
        sample_frequency = Counter(sample for sample, _ in sample_data)
        # Print sorted results by frequency and include energy
        if prn:
            print("\nSorted samples by frequency and energy:")
            for solution, freq in sample_frequency.most_common():
                energy = next(energy for sample, energy in sample_data if sample == solution)
                print(f"Sample: {solution}, Frequency: {freq}, Energy: {energy:+.2f}")
        return sample_data, sample_frequency

    def check1(self, a):
        N, M, _, _ = self.get_param()
        b = np.array(a).reshape(M, M)
        for i in range(M):
            s = 0
            for j in range(M):
                s += b[i][j]
            if s!=1:
                return False
        for j in range(M):
            s = 0
            for i in range(M):
                s += b[i][j]
            if s!=1:
                return False
        return True

    def check2(self, a):
        N, M, S, _ = self.get_param()
        b = np.array(a).reshape(N, N)
        for i in range(N):
            s = 0
            for j in range(N):
                s += b[i][j]
            if s!= S:
                return False
        #
        for j in range(N):
            s = 0
            for i in range(N):
                s += b[i][j]
            if s!= S:
                return False
        #
        '''
        s = 0
        for i in range(N):
            for j in range(N):
                if i==j:
                    s += b[i][j]
        if s!=S:
            return False
        #
        s = 0
        for i in range(N):
            k = N-i-1
            s += b[i][k]
        if s!=S:
            return False
        #
        '''
        # 右下がりの対角要素の和はS？
        s = 0
        for i in range(N):
            for j in range(N):
                if i == j:
                    s += b[i][j]
        if s != S:
            #if debug:
            #    print(f'!: 右下がりの対角要素の総和＝{s}!={S}')
            return False
        # 右上がりの対角要素の和はS?
        s = 0
        for i in range(N):
            k = N - i - 1
            s += b[i][k]
        if s != S:
            #if debug:
            #    print(f'!: 右上がりの対角要素の総和＝{s}!={S}')
            return False
        #
        return True

    def decode(self, a):
        N, M, _, _ = self.get_param()
        b = np.array(a).reshape(M, M)
        mat = []
        for i in range(M):
            for j in range(M):
                if b[i][j]==1:
                    mat.append(j+1)
        return mat

if __name__ == '__main__':
    N = 3
    mc = MagicCircle(N)
    lagrange1 = 10.0
    lagrange2 = 10.0
    lagrange3 =  1.0
    Q = mc.f(lagrange1, lagrange2, lagrange3)
    num_reads = 1000
    sampleset = mc.solv(Q, num_reads)
    #ans = mc.result(sampleset)
    #print(*ans, sep='\n')
    #
    for sample in sampleset.record['sample']:
        if mc.check1(sample):
            a = mc.decode(sample)
            if mc.check2(a):
                print(np.array(a).reshape(N, N))
                print()
