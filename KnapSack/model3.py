from knapsack import MODEL1

class MODEL3(MODEL1):
    def __init__(self):
        super().__init__()
        self.N = 5
        self.C = 12.0
        self.M = 3
        self.c = [3.0, 4.0, 6.0, 1.0, 5.0]
        self.v = [6.0, 7.0, 8.0, 1.0, 4.0]
        self.L1 = 1.0
        self.L2 = self.L1 / 9.0
        self.Q = defaultdict(lambda: 0.0)

    def cost(self, Q):
        for n1 in range(self.M):
            for n2 in range(n1+1, self.M):
                Q[(n1+self.N, n2+self.N)] += 2.0 * self.L1 * 2**n1 * 2**n2
        for a in range(self.N - 1):
            for b in range(a + 1, self.N):
                Q[(a, b)] += 2.0 * self.L1 * self.c[a] * self.c[b]
        for n in range(self.M+1):
            for a in range(self.N):
                Q[(n + self.N, a)] -= 2.0 * self.L1 * 2**n * self.c[a]
        for a in range(self.N):
            Q[(a, a)] += self.L1 * self.c[a] * self.c[a]
        for n in range(self.M + 1):
            Q[(n + self.N, n + self.N)] += self.L1 * 2**(2*n)

    def hamiltonian(self, Q):
        self.cost(Q)
        self.value(Q)

if __name__ == '__main__':
    model = MODEL1()
    Q = defaultdict(lambda: 0.0)
    model.hamiltonian(Q)
    model.print_QUBO(Q)
    sampleset = model.sample(Q, num_reads=100)
    model.evaluate(sampleset)