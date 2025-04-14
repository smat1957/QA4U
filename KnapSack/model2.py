from knapsack import MODEL1
from collections import defaultdict, Counter

class MODEL2(MODEL1):
    def __init__(self):
        super().__init__()
        self.L1 = 1.0
        self.L2 = self.L1 / 9.0
        self.L3 = self.L1 / 0.2

    def slack_unique(self, Q, L1):
        for k in range(self.C):
            Q[(k+self.N, k+self.N)] -= L1
        for k in range(self.C - 1):
            for l in range(k + 1, self.C):
                Q[(k+self.N, l+self.N)] += 2.0 * L1

    def slack_sum(self, Q, L1):
        for k in range(self.C):
            Q[(k+self.N, k+self.N)] += L1 * k**2
        for k in range(self.C - 1):
            for l in range(k + 1, self.C):
                Q[(k+self.N, l+self.N)] += 2.0 * L1 * k * l
        for a in range(self.N):
            Q[(a, a)] += L1 * self.c[a] * self.c[a]
        for a in range(self.N - 1):
            for b in range(a + 1, self.N):
                Q[(a, b)] += 2.0 * L1 * self.c[a] * self.c[b]
        for a in range(self.N - 1):
            for b in range(a + 1, self.N):
                Q[(a, b)] += 2.0 * L1 * self.c[a] * self.c[b]
        for k in range(self.C):
            for a in range(self.N):
                Q[(k+self.N, a)] -= 2.0 * L1 * k * self.c[a]

    def cost2(self, Q, L1):
        self.slack_unique(Q, L1)
        self.slack_sum(Q, L1)

    def hamiltonian(self, Q):
        self.cost(Q, self.L3)
        self.cost2(Q, self.L1)
        self.value(Q, self.L2)

if __name__ == "__main__":
    model = MODEL2()
    Q = defaultdict(lambda: 0.0)
    model.hamiltonian(Q)
    model.print_QUBO(Q)
    sampleset = model.sample(Q, num_reads=10)
    model.evaluate(sampleset)