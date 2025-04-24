from openjij import SASampler, SQASampler
from collections import defaultdict, Counter
import numpy as np

class TSP:
    def __init__(self, city, cost_matrix):
        samplers = [SASampler(), SQASampler()]
        self.sampler = samplers[0]
        self.cities = len(city)
        self.cost_matrix = np.array(cost_matrix)

    def gen_qubo1(self, cities):
        qubo_size = cities * cities
        Q1 = np.zeros((qubo_size, qubo_size))
        # u, v は訪れる都市。i, jは巡回の順番
        indices = [(u, v, i, j) for u in range(cities) for v in range(cities) for i in range(cities) for j in range(cities)]
        for u,v,i,j in indices:
            ui = u * cities + i
            vj = v * cities + j
            if ui>vj:   # 上三角だけ
                continue
            if ui==vj:  # 対角要素 \sum_\alpha\sum_i
                Q1[(ui, vj)] -= 2
            if u==v and i!=j:   # 都市が同じ(u==v)でタイミングが異なる(i!=j)
                Q1[(ui, vj)] += 2
            if u<v and i==j:    # 同一タイミング(i==j)で都市が異なる(u<v)
                Q1[(ui, vj)] += 2
        return Q1

    def gen_qubo2(self, cities, cost_matrix):
        qubo_size = cities * cities
        Q2 = np.zeros((qubo_size, qubo_size))
        # u, v は訪れる都市。i, jは巡回の順番
        indices = [(u, v, i, j) for u in range(cities) for v in range(cities) for i in range(cities) for j in range(cities)]
        for u,v,i,j in indices:
            ui = u * cities + i
            vj = v * cities + j
            k = abs(i - j)
            if ui>vj:   # 上三角だけ
                continue
            if (k == 1 or k == (cities - 1)) and u < v:   # 隣り合う都市順なら
                for r in range(len(cost_matrix)):
                    if cost_matrix[r][0] == u and cost_matrix[r][1] == v:
                        Q2[ui][vj] += cost_matrix[r][2]  # 都市のuとvの間のコスト
        return Q2

    def gen_qubo(self, lagrange1=1.0, lagrange2=1.0):
        Q1 = self.gen_qubo1(self.cities)
        Q2 = self.gen_qubo2(self.cities, self.cost_matrix)
        Q = lagrange1 * Q1 + lagrange2 * Q2
        return Q

    def solv(self, Q, num_reads=1):
        response = self.sampler.sample_qubo(Q, num_reads=num_reads)
        #sample = response.first.sample
        return response

    def result(self, sample_frequency):
        solved = []
        for solution in sample_frequency:
            if not self.check(np.array(solution).reshape(cities, cities)):
                continue
            else:
                solved.append(solution)
        ans = []
        min_cost = cost_matrix[0][2] * 100
        for item in solved:
            jyun = []
            w = np.array(item).reshape(cities, cities)
            for row in range(cities):
                for clmn in range(cities):
                    if w[row][clmn] == 1:
                        jyun.append(city[clmn])
            cost = u = v = 0
            for i, c in enumerate(jyun):
                u = city.index(jyun[i])
                if i == len(jyun) - 1:
                    v = city.index(jyun[0])
                else:
                    v = city.index(jyun[i + 1])
                for r in range(len(cost_matrix)):
                    if cost_matrix[r][0] == u and cost_matrix[r][1] == v:
                        cost += cost_matrix[r][2]  # 都市のuとvの間のコスト
                        break
            sol = (jyun, cost)
            if cost<min_cost:
                min_cost = cost
            ans.append(sol)
        return ans, min_cost

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

    def check(self, w):
        for row in w:
            sum=0
            for s in row:
                sum += s
            if sum!=1:
                return False
        wt = w.T
        for row in wt:
            sum = 0
            for s in row:
                sum += s
            if sum!=1:
                return False
        return True

if __name__ == '__main__':
    city=["A", "B", "C", "D", "E"]
    cost = np.array([
    [0.0, 3.0, 4.0, 2.0, 7.0],
    [3.0, 0.0, 4.0, 6.0, 3.0],
    [4.0, 4.0, 0.0, 5.0, 8.0],
    [2.0, 6.0, 5.0, 0.0, 6.0],
    [7.0, 3.0, 8.0, 6.0, 0.0]])
    cost_matrix = []
    for i, row in enumerate(cost):
        for j, column in enumerate(cost.T):
            if i!=j:
                row1 = [i, j, cost[i][j]]
                cost_matrix.append(row1)
    tsp = TSP(city, cost_matrix)
    lagrange2=1/15
    Q = tsp.gen_qubo(lagrange2=lagrange2)
    sampleset = tsp.solv(Q, num_reads=100)
    sample_data, sample_frequency = tsp.evaluate(sampleset, prn=False)
    answer, min_cost = tsp.result(sample_frequency)
    for ans, cost in answer:
        if cost==min_cost:
            print(ans, cost)
