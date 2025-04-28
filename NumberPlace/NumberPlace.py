from openjij import SASampler, SQASampler
from collections import defaultdict, Counter
import numpy as np

class NumberPlace:
    def __init__(self, M=2, FileN='data.txt'):
        self.M = M
        self.N = M * M
        S = 0
        for i in range(1, self.N+1):
            S += i
        self.S = S
        with open(FileN, 'r') as f:
            self.required = f.read().splitlines()
        self.idx = {}
        k = 0
        for i in range(self.N):
            for j in range(self.N):
                for n in range(self.N):
                    self.idx[(i,j,n)] = k
                    k += 1
        samplers = [SASampler(), SQASampler()]
        self.sampler = samplers[0]

    def get_param(self):
        return self.N, self.M, self.S, self.idx

    def block_ij(self):
        i0j0 = []
        for i in range(self.M):
            i0j0.append(self.M*i)
        return i0j0

    def sub1(self, i, j, L, Q):
        N, _, _, idx = self.get_param()
        for n1 in range(N):
            Q[(idx[(i, j, n1)], idx[(i, j, n1)])] -= 2.0 * L
            for n2 in range(N):
                Q[(idx[(i, j, n1)], idx[(i, j, n2)])] += 1.0 * L

    def f1(self, L, Q):
        N, _, _, _ = self.get_param()
        for i in range(N):
            for j in range(N):
                self.sub1(i, j, L, Q)
        return Q

    def sub2R(self, i, n, L, Q):
        N, _, _, idx = self.get_param()
        for j1 in range(N):
            Q[(idx[(i, j1, n)], idx[(i, j1, n)])] -= 2.0 * L
            for j2 in range(N):
                Q[(idx[(i, j1, n)], idx[(i, j2, n)])] += 1.0 * L

    def sub2C(self, j, n, L, Q):
        N, _, _, idx = self.get_param()
        for i1 in range(N):
            Q[(idx[(i1, j, n)], idx[(i1, j, n)])] -= 2.0 * L
            for i2 in range(N):
                Q[(idx[(i1, j, n)], idx[(i2, j, n)])] += 1.0 * L

    def f2(self, L, Q):
        N, _, _, _ = self.get_param()
        for i in range(N):
            for n in range(N):
                self.sub2R(i, n, L, Q)
        for j in range(N):
            for n in range(N):
                self.sub2C(j, n, L, Q)
        return Q

    def sub3(self, i0, j0, n, L, Q):
        N, M, _, idx = self.get_param()
        for x1 in range(M):
            for y1 in range(M):
                Q[(idx[(i0 + x1, j0 + y1, n)], idx[(i0 + x1, j0 + y1, n)])] -= 2.0 * L
                for x2 in range(M):
                    for y2 in range(M):
                        Q[(idx[(i0 + x1, j0 + y1, n)], idx[(i0 + x2, j0 + y2, n)])] += 1.0 * L

    def f3(self, L, Q):
        N, _, _, idx = self.get_param()
        i0j0 = self.block_ij()
        for i0 in i0j0:
            for j0 in i0j0:
                for n in range(N):
                    self.sub3(i0, j0, n, L, Q)
        return Q

    def sub4R(self, i, L, Q):
        N, _, S, idx = self.get_param()
        for j1 in range(N):
            for n1 in range(N):
                Q[(idx[(i, j1, n1)], idx[(i, j1, n1)])] -= 2.0 * (n1+1) * S * L
                for j2 in range(N):
                    for n2 in range(N):
                        Q[(idx[(i, j1, n1)], idx[(i, j2, n2)])] += (n1+1) * (n2+1) * L

    def sub4C(self, j, L, Q):
        N, _, S, idx = self.get_param()
        for i1 in range(N):
            for n1 in range(N):
                Q[(idx[(i1, j, n1)], idx[(i1, j, n1)])] -= 2.0 * (n1+1) * S * L
                for i2 in range(N):
                    for n2 in range(N):
                        Q[(idx[(i1, j, n1)], idx[(i2, j, n2)])] += (n1+1) * (n2+1) * L

    def f4(self, L, Q):
        N, _, _, _ = self.get_param()
        for i in range(N):
            self.sub4R(i, L, Q)
        for j in range(N):
            self.sub4C(j, L, Q)
        return Q

    def sub5(self, i0, j0, L, Q):
        N, M, S, idx = self.get_param()
        for x1 in range(M):
            for y1 in range(M):
                for n1 in range(N):
                    Q[(idx[(i0+x1, j0+y1, n1)], idx[(i0+x1, j0+y1, n1)])] -= 2.0 * (n1+1) * S * L
                    for x2 in range(M):
                        for y2 in range(M):
                            for n2 in range(N):
                                Q[(idx[(i0+x1, j0+y1, n1)], idx[(i0+x2, j0+y2, n2)])] += (n1+1) * (n2+1) * L

    def f5(self, L, Q):
        i0j0 = self.block_ij()
        for i0 in i0j0:
            for j0 in i0j0:
                self.sub5(i0, j0, L, Q)
        return Q

    def f6(self, I, J, X, L, Q):
        N, _, _, idx = self.get_param()
        for n1 in range(N):
            Q[(idx[(I, J, n1)], idx[(I, J, n1)])] -= 2 * (n1+1) * X * L
            for n2 in range(N):
                Q[(idx[(I, J, n1)], idx[(I, J, n2)])] += (n1+1) * (n2+1) * L
        return Q

    def f(self, lagrange1=1.0, lagrange2=1.0, lagrange3=1.0, lagrange4=1.0):
        Q = defaultdict(lambda: 0)
        _ = self.f1(lagrange1, Q)
        _ = self.f2(lagrange2, Q)
        _ = self.f3(lagrange2, Q)
        if 0.0 < lagrange3:
            _ = self.f4(lagrange3, Q)
            _ = self.f5(lagrange3, Q)
        for a in self.required:
            IJX = a.split(',')
            _ = self.f6(int(IJX[0]), int(IJX[1]), int(IJX[2]), lagrange4, Q)
        return Q

    def solv(self, Q, num_reads=1):
        sampleset = self.sampler.sample_qubo(Q, num_reads=num_reads)
        return sampleset

    def result(self, sampleset):
        N, _, _, idx = self.get_param()
        result = [i for i in sampleset.first[0].values()]
        ans = [[None] * N for _ in range(N)]
        for i in range(N):
            for j in range(N):
                for n in range(N):
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

    def check1(self, a, debug=False):
        N, M, _, _ = self.get_param()
        b = np.array(a).reshape(N*N, N)
        # 各セルに数値は1つ？
        for i in range(N*N):
            s = 0
            for n in range(N):
                s += b[i][n]
            if s != 1:
                if debug:
                    print(f'!: セルの中の数値が1つでない{i:3d}:{b[i]}')
                return False
        # 各ブロックに重複する数値はない？
        i0j0 = self.block_ij()
        for i in i0j0:
            for j in i0j0:
                for n in range(N):
                    s = 0
                    for x in range(M):
                        for y in range(M):
                            bidx = (i+x)*N + j+y
                            s += b[bidx][n]
                    if s != 1:
                        if debug:
                            print(f'!: ブロック内で数値が重複')
                        return False
        #
        for n in range(N):
            # 各行に重複する数値はない？
            for i in range(N):
                s = 0
                for j in range(N):
                    bidx = i * N + j
                    s += b[bidx][n]
                if s != 1:
                    if debug:
                        print(f'!: 行で数値が重複')
                    return False
            # 各列に重複する数値はない？
            for j in range(N):
                s = 0
                for i in range(N):
                    bidx = i * N + j
                    s += b[bidx][n]
                if s != 1:
                    if debug:
                        print(f'!: 列で数値が重複')
                    return False
        #
        return True

    def check2(self, a, debug=False):
        N, M, S, _ = self.get_param()
        b = np.array(a).reshape(N, N)
        # 既定値は正しい？
        for a in self.required:
            IJX = a.split(',')
            if b[int(IJX[0])][int(IJX[1])]!=int(IJX[2]):
                if debug:
                    print(f'!: 既定値が違う:({IJX[0]},{IJX[1]}){IJX[2]}!={b[int(IJX[0])][int(IJX[1])]}')
                return False
        # 各行の数値の和はS？
        for i in range(N):
            s = 0
            for j in range(N):
                s += b[i][j]
            if s != S:
                if debug:
                    print(f'!: 行の総和＝{s}!={S}')
                return False
        # 各列の数値の和はS？
        for j in range(N):
            s = 0
            for i in range(N):
                s += b[i][j]
            if s != S:
                if debug:
                    print(f'!: 列の総和＝{s}!={S}')
                return False
        # 各ブロックの数値の和はS？
        i0j0 = self.block_ij()
        for i in i0j0:
            for j in i0j0:
                s = 0
                for x in range(M):
                    for y in range(M):
                        #print(i+x,j+y)
                        s += b[i+x][j+y]
                if s != S:
                    if debug:
                        print(f'!: ブロック内の総和＝{s}!={S}')
                    return False
        #
        return True

    def decode(self, a):
        N, M, _, _ = self.get_param()
        b = np.array(a).reshape(N**2, N)
        mat = []
        for v in b:
            num = 0
            for i, u in enumerate(v):
                if u==1:
                    num = i+1
            mat.append(num)
        return mat

    def print_shape(self):
        for i in range(self.N):
            print(f'{i}:', end='\t')
            for j in range(self.N):
                for a in self.required:
                    IJX = a.split(',')
                    if i==int(IJX[0]) and j==int(IJX[1]):
                        print(int(IJX[2]), end=' ')
                        break
                else:
                    print('_', end=' ')
            print()

if __name__ == '__main__':
    KiteiF = 'dataA250429.txt'
    M = 3
    sudoku = NumberPlace(M, KiteiF)
    sudoku.print_shape()
    #lagrange1 = 40.0      # 数値に重複なし
    #lagrange2 =  5.4      # 行、列、ブロック、で重複なし
    #lagrange3 =  0.0      # 和はS
    #lagrange4 =  5.1      # 既定セル
    lagrange1 = 1.0        # 数値に重複なし
    lagrange2 = lagrange1 * 0.653       # 行、列、ブロック、で重複なし 0.018
    lagrange3 = lagrange1 * 0.0         # 和はS
    lagrange4 = lagrange1 * 0.05        # 既定セル
    Q = sudoku.f(lagrange1, lagrange2, lagrange3, lagrange4)
    num_reads = 100
    sampleset = sudoku.solv(Q, num_reads)
    ans = sudoku.result(sampleset)
    print(*ans, sep='\n')
    #
    debug = True
    for sample in sampleset.record['sample']:
        if sudoku.check1(sample, debug):
            if debug: print('check1 Passed!')
            a = sudoku.decode(sample)
            if sudoku.check2(a, debug):
                if debug: print('check2 Passed!')
                print(np.array(a).reshape(M*M, M*M))
                print()
