from openjij import SASampler, SQASampler
from collections import defaultdict, Counter

class MODEL1:
    def __init__(self):
        self.N = 5
        self.C = 12
        self.c = [3.0, 4.0, 6.0, 1.0, 5.0]
        self.v = [6.0, 7.0, 8.0, 1.0, 4.0]
        self.L1 = 1.0
        self.L2 = self.L1 / 9.0
        self.Samplers = [SASampler, SQASampler]
        self.sampler = self.Samplers[1]()

    def cost(self, Q, L1):
        for a in range(self.N):
            Q[(a, a)] += L1 * self.c[a] * (self.c[a] - 2.0 * self.C)
        for a in range(self.N - 1):
            for b in range(a + 1, self.N):
                Q[(a, b)] += 2.0 * L1 * self.c[a] * self.c[b]

    def value(self, Q, L2):
        for a in range(self.N):
            Q[(a, a)] -= L2 * self.v[a]

    def hamiltonian(self, Q):
        self.cost(Q, self.L1)
        self.value(Q, self.L2)

    def print_QUBO(self, Q):
        # Print QUBO matrix (for debugging)
        for i in range(self.N):
            for j in range(self.N):
                print("{:7.1f}".format(Q[(i, j)]), end=" ")
            print()

    def sample(self, Q, num_reads=100):
        # Perform sampling
        sampleset = self.sampler.sample_qubo(Q, num_reads=num_reads)
        return sampleset

    def evaluate(self, sampleset):
        # Extract sample solutions, energies, and sort them by frequency
        samples = sampleset.record['sample']
        energies = sampleset.record['energy']
        # Combine solutions and corresponding energies
        sample_data = [(tuple(sample), energy) for sample, energy in zip(samples, energies)]
        # Sort the results by appearance frequency and then energy
        sample_frequency = Counter(sample for sample, _ in sample_data)
        # Print sorted results by frequency and include energy
        print("\nSorted samples by frequency and energy:")
        for solution, freq in sample_frequency.most_common():
            energy = next(energy for sample, energy in sample_data if sample == solution)
            print(f"Sample: {solution}, Frequency: {freq}, Energy: {energy:+.2f}")
        # Evaluate each solution
        print("\nEvaluation of solutions:")
        for nth, (solution, freq) in enumerate(sample_frequency.most_common(), 1):
            value = volume = 0.0
            for i, bit in enumerate(solution[:self.N]):  # Consider only the first N bits for w and c
                if bit == 1:
                    volume += self.c[i]
                    value += self.v[i]
            if volume <= self.C:
                energy = next(energy for sample, energy in sample_data if sample == solution)
                print(f"[{nth:2d}] Frequency={freq}, Energy={energy:+.2f},\n Solution={solution}", end="\t")
                print(f": value={value:4.1f}, cost={volume:4.1f}")

if __name__ == "__main__":
    model = MODEL1()
    Q = defaultdict(lambda: 0.0)
    model.hamiltonian(Q)
    model.print_QUBO(Q)
    sampleset = model.sample(Q, num_reads=100)
    model.evaluate(sampleset)