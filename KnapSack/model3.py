from knapsack import MODEL1
from collections import defaultdict, Counter

class MODEL3(MODEL1):
    def __init__(self):
        super().__init__()
        self.M = 3
        self.L1 = 1.0
        self.L2 = self.L1 / 9.0
        self.L3 = self.L1 / 100.0

    def cost3(self, Q, L1):
        for n1 in range(self.M):
            for n2 in range(n1+1, self.M+1):
                Q[(n1+self.N, n2+self.N)] += 2.0 * L1 * 2**n1 * 2**n2
        for a in range(self.N - 1):
            for b in range(a + 1, self.N):
                Q[(a, b)] += 2.0 * L1 * self.c[a] * self.c[b]
        for n in range(self.M+1):
            for a in range(self.N):
                Q[(n + self.N, a)] -= 2.0 * L1 * 2**n * self.c[a]
        for a in range(self.N):
            Q[(a, a)] += L1 * self.c[a] * self.c[a]
        for n in range(self.M + 1):
            Q[(n + self.N, n + self.N)] += L1 * 2**(2*n)

    def hamiltonian(self, Q):
        self.cost(Q, self.L3)
        self.cost3(Q, self.L1)
        self.value(Q, self.L2)

if __name__ == '__main__':
    model = MODEL3()
    Q = defaultdict(lambda: 0.0)
    model.hamiltonian(Q)
    model.print_QUBO(Q)
    sampleset = model.sample(Q, num_reads=10)
    model.evaluate(sampleset)