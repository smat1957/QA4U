from knapsack import MODEL1

class MODEL2(MODEL1):
    def __init__(self):
        super().__init__()
        self.N = 5
        self.C = 12.0
        self.c = [3.0, 4.0, 6.0, 1.0, 5.0]
        self.v = [6.0, 7.0, 8.0, 1.0, 4.0]
        self.L1 = 1.0
        self.L2 = self.L1 / 9.0
        self.Q = defaultdict(lambda: 0.0)

    def slack_unique(self, Q):
        for k in range(self.C):
            Q[(k+self.N, k+self.N)] -= self.L1
        for k in range(self.C - 1):
            for l in range(k + 1, self.C):
                Q[(k+self.N, l+self.N)] += 2.0 * self.L1

    def slack_sum(self, Q):
        for k in range(self.C):
            Q[(k+self.N, k+self.N)] += self.L1 * k**2
        for k in range(self.C - 1):
            for l in range(k + 1, self.C):
                Q[(k+self.N, l+self.N)] += 2.0 * self.L1 * k * l
        for a in range(self.N):
            Q[(a, a)] += self.L1 * self.c[a] * self.c[a]
        for a in range(self.N - 1):
            for b in range(a + 1, self.N):
                Q[(a, b)] += 2.0 * self.L1 * self.c[a] * self.c[b]
        for a in range(self.N - 1):
            for b in range(a + 1, self.N):
                Q[(a, b)] += 2.0 * self.L1 * self.c[a] * self.c[b]
        for k in range(self.C):
            for a in range(self.N):
                Q[(k+self.N, a)] -= 2.0 * self.L1 * k * self.c[a]

    def cost(self, Q):
        self.slack_unique(Q)
        self.slack_sum(Q)

    def hamiltonian(self, Q):
        self.cost(Q)
        self.value(Q)

if __name__ == "__main__":
    model = MODEL1()
    Q = defaultdict(lambda: 0.0)
    model.hamiltonian(Q)
    model.print_QUBO(Q)
    sampleset = model.sample(Q, num_reads=100)
    model.evaluate(sampleset)