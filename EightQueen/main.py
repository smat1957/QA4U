from collections import defaultdict
import numpy as np
from dwave.optimization.symbols import Sum
from dwave.system import DWaveSampler, EmbeddingComposite
from openjij import SASampler, SQASampler
import time

rotate90 = lambda A: [list(x)[::-1] for x in zip(*A)]  # 90度回転用関数
transpose = lambda A: [list(x) for x in zip(*A)]  # 転置用関数

class EightQueen:
    def __init__(self):
        self.N = 8  # 盤面の大きさはNxN
        self.S = 1.0  # 各列の総和
        self.l1 = 2.0  # 罰金項の強さ（各行、各列の総和は等しい）
        self.l2 = 1.5  # 罰金項の強さ（斜めの総和は0または1）
        self.l3 = 1.0   # 罰金項の強さ（盤面上はN個）
        self.num_reads = 10000   # アニーリングを実行する回数
        self.token = 'XXXX'  # API token(個人のものを使用)
        self.solver = 'Advantage_system6.4'  # 量子アニーリングマシン
        self.machine = False
        Sampler = [SASampler(), SQASampler()]
        self.sampler = Sampler[0]
        self.ij_to_idx = {}

    def myindex(self):
        Q = defaultdict(lambda: 0)  # Q_i,j  (i, j)に入れる値
        # x_i,jの通し番号を記録
        ij_to_idx = {}
        idx = 0
        for i in range(self.N):
            for j in range(self.N):
                ij_to_idx[(i, j)] = idx
                idx += 1
        self.ij_to_idx = ij_to_idx
        return Q, ij_to_idx

    def constraint(self, Q, ij_to_idx):
        Q = self.constraint_row(Q, ij_to_idx)
        Q = self.constraint_clmn(Q, ij_to_idx)
        Q = self.constraint_diagonal(Q, ij_to_idx)
        Q = self.constraint_N(Q, ij_to_idx)
        return Q

    def constraint_row(self, Q,  ij_to_idx):
        # 各行の総和をSに制限(f1)
        for i in range(self.N):
            for j1 in range(self.N):
                for j2 in range(self.N):
                    Q[(ij_to_idx[(i, j1)], ij_to_idx[(i, j2)])] += self.l1
                Q[(ij_to_idx[(i, j1)], ij_to_idx[(i, j1)])] -= 2 * self.S * self.l1
        return Q

    def constraint_clmn(self, Q,  ij_to_idx):
        # 各列の総和をSに制限(f2)
        for j in range(self.N):
            for i1 in range(self.N):
                for i2 in range(self.N):
                    Q[(ij_to_idx[(i1, j)], ij_to_idx[(i2, j)])] += self.l1
                Q[(ij_to_idx[(i1, j)], ij_to_idx[(i1, j)])] -= 2 * self.S * self.l1
        return Q

    def constraint_diagonal(self, Q, ij_to_idx):
        # 斜めの各列の総和は0または1
        # 左上から右下の斜め制約 (i - j が等しい)
        for k in range(-self.N + 1, self.N):  # 有効なダイアゴナルの範囲
            diagonal_indices = [(i, j) for i in range(self.N) for j in range(self.N) if i - j == k]
            for idx1 in diagonal_indices:
                for idx2 in diagonal_indices:
                    Q[(ij_to_idx[idx1], ij_to_idx[idx2])] += self.l2
                Q[(ij_to_idx[idx1], ij_to_idx[idx1])] -= 2 * self.S * self.l2

        # 右上から左下の斜め制約 (i + j が等しい)
        for k in range(2 * self.N - 1):  # 有効なダイアゴナルの範囲
            diagonal_indices = [(i, j) for i in range(self.N) for j in range(self.N) if i + j == k]
            for idx1 in diagonal_indices:
                for idx2 in diagonal_indices:
                    Q[(ij_to_idx[idx1], ij_to_idx[idx2])] += self.l2
                Q[(ij_to_idx[idx1], ij_to_idx[idx1])] -= 2 * self.S * self.l2
        return Q
        
    def constraint_N(self, Q, ij_to_idx):
        for i1 in range(self.N):
            for j1 in range(self.N):
                for i2 in range(self.N):
                    for j2 in range(self.N):
                        Q[(ij_to_idx[(i1, j1)], ij_to_idx[(i2, j2)])] += self.l3
                Q[(ij_to_idx[(i1, j1)], ij_to_idx[(i1, j1)])] -= 2 * self.N * self.l3
        return Q
        
    def annealing(self, Q):
        if self.machine:
            # 量子アニーリングの実行
            endpoint = 'https://cloud.dwavesys.com/sapi/'
            dw_sampler = DWaveSampler(solver=self.solver, token=self.token, endpoint=endpoint)
            sampler = EmbeddingComposite(dw_sampler)
            sampleset = sampler.sample_qubo(Q, num_reads=self.num_reads)
        else:
            # 焼きなまし法の実行
            sampler = self.sampler
            sampleset = sampler.sample_qubo(Q, num_reads=self.num_reads)
        return sampleset

    def gen_mat(self, sset):
        mat = []
        for i in range(self.N):
            w = []
            for j in range(self.N):
                w.append(sset[(self.ij_to_idx[(i, j)])])
            mat.append(w)
        return mat

    def same_p(self, sset1, sset2):
        s1 = np.array(sset1).flatten()
        s2 = np.array(sset2).flatten()
        if np.array_equal(s1, s2):
            return True
        return False

    def check_row_clmn(self, mat):
        Sum = self.S
        for i in range(self.N):
            s = 0.0
            for j in range(self.N):
                s += mat[i][j]
            if Sum < s:
                return False
        return True

    def check_diagonal(self, mat):
        Sum = self.S
        # 左上から右下の斜め (i - j が等しい)
        for k in range(-self.N + 1, self.N):  # 有効なダイアゴナルの範囲
            diagonal_indices = [(i, j) for i in range(self.N) for j in range(self.N) if i - j == k]
            s = 0
            for d in diagonal_indices:
                s += mat[d[0]][d[1]]
            if Sum < s:
                return False
        # 右上から左下の斜め (i + j が等しい)
        for k in range(2 * self.N - 1):  # 有効なダイアゴナルの範囲
            diagonal_indices = [(i, j) for i in range(self.N) for j in range(self.N) if i + j == k]
            s = 0
            for d in diagonal_indices:
                s += mat[d[0]][d[1]]
            if Sum < s:
                return False
        return True

    def check_valid(self, sampleset):
        removelist = []
        for nth, sset in enumerate(sampleset):
            # Convert the solution into the 2D matrix form
            mat = self.gen_mat(sset)  # Properly generate the matrix form
            # Conduct various checks (rows, columns, diagonals)
            ck1 = self.check_row_clmn(mat)  # Check rows
            if not ck1:
                removelist.append(nth)
                continue
            ck2 = self.check_row_clmn(transpose(mat))  # Check columns
            if not ck2:
                removelist.append(nth)
                continue
            ck3 = self.check_diagonal(mat)  # Check diagonals
            if not ck3:
                removelist.append(nth)
                continue
        return self.remove_from_sampleset(sampleset, removelist)

    def check_duplicate(self, sampleset):
        # Ensure no duplicates due to symmetry or rotations
        removelist = []
        for nth1, sset1 in enumerate(sampleset):
            if nth1 in removelist:
                continue
            sets1 = self.variable_mat(sset1)
            for nth2, sset2 in enumerate(sampleset):
                if nth1==nth2 or nth2 in removelist:
                    continue
                sets2 = self.variable_mat(sset2)
                for elem1 in sets1:
                    for elem2 in sets2:
                        if self.same_p(elem1, elem2):
                            if nth2 not in removelist:
                                removelist.append(nth2)
                                break
                else:
                    continue
        return self.remove_from_sampleset(sampleset, removelist)

    def remove_from_sampleset(self, sampleset, removelist):
        sampleset1 = []
        for nth, sset in enumerate(sampleset):
            if nth not in removelist:
                sampleset1.append(sset)
        return sampleset1

    def variable_mat(self, sset):
        sets = []
        set0 = self.gen_mat(sset)  # Convert to matrix form
        set1 = rotate90(set0)    # rotate 90
        set2 = rotate90(set1)   # rotate 180
        set3 = rotate90(set2)   # rotate 270
        set4 = np.fliplr(np.array(set)).copy().tolist()
        set5 = np.flipud(np.array(set)).copy().tolist()
        sets = [set0, set1, set2, set3, set4, set5]
        return sets

    def check(self, sampleset):
        sampleset1 = self.check_valid(sampleset)
        tm1 = time.time()
        print("valid   =", len(sampleset1))
        sampleset2 = self.check_duplicate(sampleset1)
        tm2 = time.time()
        print("checked =", len(sampleset2))
        return sampleset2, tm1, tm2

    def result(self, resultset):
        for nth, sset in enumerate(resultset):
            mat = self.gen_mat(sset)
            print(nth+1)
            for i in range(self.N):
                print(mat[i])
            print()

    def result_check_do(self, resultset):
        for nth, sset in enumerate(resultset):
            sset = self.gen_mat(sset)  # 行列の形に変換
            set1 = rotate90(sset)
            set2 = rotate90(set1)
            set3 = rotate90(set2)
            set4 = (np.array(sset)[:, ::-1]).tolist()
            set5 = (np.array(sset).T[:, ::-1].T).tolist()
            print(nth+1)
            for i in range(self.N):
                print(sset[i], set1[i], set2[i], set3[i], set4[i], set5[i])
            print()

if __name__ == '__main__':
    start = time.time()
    eq = EightQueen()
    Q, ij2idx = eq.myindex()
    Q = eq.constraint(Q, ij2idx)
    ckpt1 = time.time()
    sampleset = eq.annealing(Q)
    ckpt2 = time.time()
    print("annealed=",len(sampleset))
    resultset, ckpt3, ckpt4 = eq.check(sampleset)
    eq.result(resultset)
    #
    print("Prepare:{}".format(ckpt1 - start))
    print("Annealing:{}".format(ckpt2 - ckpt1))
    print("ValidateCheck:{}".format(ckpt3 - ckpt2))
    print("DuplicateCheck:{}".format(ckpt4 - ckpt3))
    print("Total:{}".format(ckpt4 - start))
