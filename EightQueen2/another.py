from __future__ import annotations
from main import EightQueen
import numpy as np
from openjij import SASampler, SQASampler
from collections import defaultdict

class AnotherEQ(EightQueen):
    def __init__(self, N=8, S=1, lambda_1=1, lambda_3=1, lambda_diag=1):
        super().__init__()
        self.N = N  # 盤面のサイズ
        self.l_1 = lambda_1
        self.l_3 = lambda_3
        self.l_diag = lambda_diag
        self.S = S
        self.ijn_to_idx = {}     # QUBOキー部のタプルとインデックスの対応
        self.samplers = [SASampler(), SQASampler()]

    def qubo_init(self):
        Q = defaultdict(lambda: 0)
        self.ijn_to_idx = {}    # QUBOキー部のタプルとインデックスの対応
        idx = 0
        for i in range(self.N):
            for j in range(self.N):
                for n in range(2):
                    self.ijn_to_idx[(i, j, n)] = idx
                    idx += 1
        return Q

    def add_row_column_constraints(self, Q):
        # 行方向の和は1
        for i in range(self.N):
            for j1 in range(self.N):
                for n1 in range(2):
                    idx = self.ijn_to_idx[(i, j1, n1)]
                    for j2 in range(self.N):
                        for n2 in range(2):
                            jdx = self.ijn_to_idx[(i, j2, n2)]
                            Q[(idx,jdx)] += n1 * n2 * self.l_1
                    Q[(idx,idx)] -= 2 * n1 * self.S * self.l_1
        # 列方向の和は1
        for j in range(self.N):
            for i1 in range(self.N):
                for n1 in range(2):
                    idx = self.ijn_to_idx[(i1, j, n1)]
                    for i2 in range(self.N):
                        for n2 in range(2):
                            jdx = self.ijn_to_idx[(i2, j, n2)]
                            Q[(idx,jdx)] += n1 * n2 * self.l_1
                    Q[(idx,idx)] -= 2 * n1 * self.S * self.l_1

    def add_diagonal_constraints(self, Q):
        """
        Add constraints to ensure that all diagonal (main and anti-diagonal, including offset diagonals)
        sums are equal to 1 in the QUBO formulation.

        Args:
            Q (dict): QUBO dictionary to store constraints.
            ijn_to_idx (dict): Index mapping for (i, j, n).
            N (int): Size of the grid (N x N).
            l_diag (float): Weight for diagonal constraints.

        Returns:
            None: Modifies Q in-place to add diagonal constraints.
        """
        # --- 主対角線およびその平行な斜め方向の制約を追加 ---
        for offset in range(-self.N + 1, self.N):  # オフセットを考慮
            involved_indices = []
            for i in range(self.N):
                j = i + offset
                if 0 <= i < self.N and 0 <= j < self.N:  # 有効な範囲
                    for n in range(2):  # n = 0 or 1
                        involved_indices.append(self.ijn_to_idx[(i, j, n)])

            # 制約のペナルティ項を QUBO に追加
            for idx1 in involved_indices:
                for idx2 in involved_indices:
                    Q[(idx1, idx2)] += self.l_diag  # 対角項
                Q[(idx1, idx1)] -= 2 * self.S * self.l_diag  # 線形項

        # --- 副対角線およびその平行な斜め方向の制約を追加 ---
        for offset in range(-self.N + 1, self.N):  # オフセットを考慮
            involved_indices = []
            for i in range(self.N):
                j = self.N - 1 - i + offset
                if 0 <= i < self.N and 0 <= j < self.N:  # 有効な範囲
                    for n in range(2):  # n = 0 or 1
                        involved_indices.append(self.ijn_to_idx[(i, j, n)])

            # 制約のペナルティ項を QUBO に追加
            for idx1 in involved_indices:
                for idx2 in involved_indices:
                    Q[(idx1, idx2)] += self.l_diag  # 対角項
                Q[(idx1, idx1)] -= 2 * self.S * self.l_diag  # 線形項

    def add_only_one_number_placed_constraints(self, Q):
        # 各マスには1つしか数値を置かない
        for i in range(self.N):
            for j in range(self.N):
                for n1 in range(2):
                    idx = self.ijn_to_idx[(i, j, n1)]
                    for n2 in range(2):
                        jdx = self.ijn_to_idx[(i, j, n2)]
                        Q[(idx, jdx)] += 1 * self.l_3
                    Q[(idx, idx)] -= 2 * self.l_3

    def generate_QUBO(self):
        Q = self.qubo_init()
        #--- 行方向と列方向の総和制約を追加 ---
        self.add_row_column_constraints(Q)
        #--- 各マスには1つしか数値を置かない ---
        self.add_only_one_number_placed_constraints(Q)
        # --- 斜め方向の総和制約を追加 ---
        self.add_diagonal_constraints(Q)
        # QUBO行列を返す
        return Q

    def solve(self, Q, sampler=0, num_reads=100):
        sampleset = self.samplers[sampler].sample_qubo(Q, num_reads=num_reads)
        # 計算の結果出力
        result_records = list(sampleset.data())
        # エネルギーの大きい順に並び替え(最後が最も小さい)
        sorted_results = sorted(result_records, key=lambda record: record.energy, reverse=True)
        # 結果を格納
        all_answers = []
        for k, record in enumerate(sorted_results):
            candidate = list(record.sample.values())
            ans = [[0] * self.N for _ in range(self.N)]  # ans = [[None] * N for _ in range(N)]
            for i in range(self.N):
                for j in range(self.N):
                    for n in range(2):
                        if candidate[self.ijn_to_idx[(i, j, n)]] == 1:
                            ans[i][j] = n
            all_answers.append((record.energy, ans))  # エネルギーと解を保存
        # 点対称、線対称、重複チェック
        non_symmetric = self.remove_symmetric_duplicates(all_answers)
        unique_results = self.filter_answers(non_symmetric)
        return unique_results
        #return all_answers

    def select_sampler(self):
        pass

    def rotate_180(self, answer):
        """
        与えられた2次元リストを180度回転させる関数
        """
        # 入力データが2次元リストであることを確認
        if not isinstance(answer, list) or not all(isinstance(row, list) for row in answer):
            raise ValueError("rotate_180 関数には2次元リストを渡す必要があります。")
        return [row[::-1] for row in answer[::-1]]

    def rotate_90(self, answer):
        """
        与えられた2次元リストを90度回転させる関数
        """
        if not isinstance(answer, list) or not all(isinstance(row, list) for row in answer):
            raise ValueError("rotate_90 関数には2次元リストを渡す必要があります。")
        # 転置して各行を逆順にすることで90度回転
        return [list(row) for row in zip(*answer[::-1])]

    def same_p(self, m, result, unique_results):
        num = []
        entry = result.flatten()
        for n, matrix in enumerate(unique_results):
            if m!=n:
                work = np.array(matrix[1]).flatten()
                if np.array_equal(entry, work):
                    num.append(n)
        return num

    def line_symmetric_duplicates(self, unique_results):
        filtered_results = []
        num0, num1, num2 = [], [], []
        for m, result in enumerate(unique_results):
            entry = np.array(result[1])[:, ::-1]    # 左右対称チェック
            a = np.array( num1 )
            b = np.array( self.same_p(m, entry, unique_results) )
            num1 = np.concatenate([a, b], axis=0)
            #
            entry = np.array(result[1]).T[:, ::-1].T    # 上下対称チェック
            a = np.array( num2 )
            b = np.array( self.same_p(m, entry, unique_results) )
            num2 = np.concatenate([a, b], axis=0)
            #
            entry = np.array(result[1])
            a = np.array( num0 )
            b = np.array( self.same_p(m, entry, unique_results) )
            num0 = np.concatenate([a, b], axis=0)
        for m, result in enumerate(unique_results):
            if m not in num0 and m not in num1 and m not in num2:
                filtered_results.append(result)
        return filtered_results

    def remove_symmetric_duplicates(self, results):
         ww = self.point_symmetric_duplicates(results)
         return self.line_symmetric_duplicates(ww)

    def point_symmetric_duplicates(self, results):
        """
        回転対称 (90度, 180度, 270度) な解を削除する関数
        """
        unique_results = []
        seen = set()
        for energy, result in results:
            # 各回転状態を計算
            rotated_90 = self.rotate_90(result)
            rotated_180 = self.rotate_90(rotated_90)  # 90度回転をさらに90度回転で180度
            rotated_270 = self.rotate_90(rotated_180)  # 180度回転をさらに90度回転で270度
            # 全方向の状態をタプル形式に変換
            result_tuple = tuple(map(tuple, result))
            rotated_90_tuple = tuple(map(tuple, rotated_90))
            rotated_180_tuple = tuple(map(tuple, rotated_180))
            rotated_270_tuple = tuple(map(tuple, rotated_270))
            # どれか1つでも既に記録されていればスキップ
            if (result_tuple not in seen and
                    rotated_90_tuple not in seen and
                    rotated_180_tuple not in seen and
                    rotated_270_tuple not in seen):
                # 重複がなければ追加
                unique_results.append((energy, result))
                seen.add(result_tuple)  # オリジナルを記録
                seen.add(rotated_90_tuple)  # 90度回転を記録
                seen.add(rotated_180_tuple)  # 180度回転を記録
                seen.add(rotated_270_tuple)  # 270度回転を記録
        return unique_results

    def check_diagonal_sums(self, matrix):
        n = matrix.shape[0]
        #print(matrix)
        # 主対角線とその平行な斜め方向のチェック
        for offset in range(-n + 1, n):  # オフセットを考慮
            #print(offset, np.diagonal(matrix, offset=offset))
            diag_sum = np.sum(np.diagonal(matrix, offset=offset))
            if diag_sum > 1:  # 条件：斜めの和が 1
                return False
        # 副対角線とその平行な斜め方向のチェック
        flipped_matrix = np.fliplr(matrix)
        for offset in range(-n + 1, n):  # オフセットを考慮
            anti_diag_sum = np.sum(np.diagonal(flipped_matrix, offset=offset))
            if anti_diag_sum > 1:  # 条件：斜めの和が 1
                return False
        return True

    def filter_answers(self, all_answers):
        filtered_answers = []
        for energy, ans in all_answers:
            ans = np.array(ans)
            # None を含む行列は除外
            if np.any(ans == None):  # Noneチェック
                continue
            # 横方向の和がすべて1
            row_sum_valid = np.all(np.sum(ans, axis=1) == 1)
            # 縦方向の和がすべて1
            col_sum_valid = np.all(np.sum(ans, axis=0) == 1)
            # 斜め方向のすべての和が1
            diagonal_sum_valid = self.check_diagonal_sums(ans)
            # 条件を満たしている場合のみ追加
            if row_sum_valid and col_sum_valid and diagonal_sum_valid:
                filtered_answers.append((energy, ans.tolist()))
        return filtered_answers

    def result(self, results):
        # 結果を表示
        print("回転対称、鏡像を除いた結果:\n")
        for idx, (energy, ans) in enumerate(results):
            print(f"\n候補 {idx + 1} (エネルギ: {energy}):")
            for row in ans:
                print(" ".join(map(str, row)))
            print("-" * 16)

if __name__ == "__main__":
    eq = AnotherEQ(N=8, S=1, lambda_1=1, lambda_3=1, lambda_diag=1)
    Q = eq.generate_QUBO()
    results = eq.solve(Q, sampler=0, num_reads=10000)
    #for result in results:
    #    print(result)
    eq.result(results)
