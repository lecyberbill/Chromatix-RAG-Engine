# -*- coding: utf-8 -*-
# [WFGY] Zone: SAFE | λ: 0.1 | Action: Implémentation de SEH V1.3 en Python pour le RAG
import math

class SehConfig:
    def __init__(self, n=60, inhib=0.10, gain=0.30, beta=0.25, coupling=0.30,
                 damping=0.10, c_target=0.05, alpha_pol=0.10, cycles=500,
                 threshold=0.5, noise_amp=0.3, allow_negative=False, max_winners=None):
        self.n = n
        self.inhib = inhib
        self.gain = gain
        self.beta = beta
        self.coupling = coupling
        self.damping = damping
        self.c_target = c_target
        self.alpha_pol = alpha_pol
        self.cycles = cycles
        self.threshold = threshold
        self.noise_amp = noise_amp
        self.allow_negative = allow_negative
        self.max_winners = max_winners

class SehRng:
    def __init__(self, seed: int):
        self.seed = seed & 0xFFFFFFFFFFFFFFFF

    def next_f64(self) -> float:
        x = self.seed
        x ^= (x >> 12) & 0xFFFFFFFFFFFFFFFF
        x ^= (x << 25) & 0xFFFFFFFFFFFFFFFF
        x ^= (x >> 27) & 0xFFFFFFFFFFFFFFFF
        self.seed = x & 0xFFFFFFFFFFFFFFFF
        v = (self.seed * 0x2545F4914F6CDD1D) & 0xFFFFFFFFFFFFFFFF
        return float(v >> 11) / float(1 << 53) * 2.0 - 1.0

class SehField:
    def __init__(self, patterns: list, config: SehConfig):
        self.z = [0.0] * config.n
        self.v = [0.0] * config.n
        self.patterns = patterns
        self.config = config

    def proj(self, j: int) -> float:
        m = self.patterns[j]
        return sum(a * b for a, b in zip(self.z, m))

    def projections(self) -> list:
        return [self.proj(j) for j in range(len(self.patterns))]

    def tick(self):
        n = self.config.n
        nb = len(self.patterns)
        dt = 0.10
        damping = self.config.damping
        coupling = self.config.coupling
        gain = self.config.gain
        beta = self.config.beta
        c_target = self.config.c_target
        inhib = self.config.inhib
        alpha_pol = self.config.alpha_pol

        projs = self.projections()
        total_sq = sum(p * p for p in projs)

        res = []
        for j in range(nb):
            p = projs[j]
            p2 = p * p
            equalise = -c_target * p
            bistable = gain * p * p2 - beta * p * p2 * p2
            other_sq = total_sq - p2
            inhibition = inhib * p * other_sq
            res.append(equalise + bistable - inhibition)

        friction = 1.0 - damping * dt
        for i in range(n):
            acc = (self.z[(i + n - 1) % n] + self.z[(i + 1) % n] - 2.0 * self.z[i]) * coupling
            rappel = sum(self.patterns[j][i] * res[j] for j in range(nb))
            self.v[i] = self.v[i] * friction + (acc + rappel) * dt

        for i in range(n):
            self.z[i] += self.v[i] * dt

        projs2 = self.projections()
        pres = sum(p * p for p in projs2)
        force_pol = alpha_pol * pres
        for i in range(n):
            mem = sum(projs2[j] * self.patterns[j][i] for j in range(nb))
            complement = self.z[i] - mem
            self.z[i] -= force_pol * complement

    def semer(self, coeffs: list, rng: SehRng):
        n = self.config.n
        bruit_amp = self.config.noise_amp
        nb = len(self.patterns)
        for i in range(n):
            val = bruit_amp * rng.next_f64()
            for j in range(nb):
                if j < len(coeffs):
                    val += coeffs[j] * self.patterns[j][i]
            self.z[i] = val
        self.v = [0.0] * n

def decide(coeffs: list, patterns: list, config: SehConfig) -> dict:
    nb = len(patterns)
    assert len(coeffs) == nb, f"coeffs.len() ({len(coeffs)}) != patterns.len() ({nb})"

    field = SehField(patterns, config)
    rng = SehRng(0xABCDEF)
    field.semer(coeffs, rng)

    early_stop_threshold = 0.95
    max_cycles = config.cycles
    cycles_used = 0
    early_stop = False

    for c in range(max_cycles):
        field.tick()
        cycles_used = c + 1

        if c > 50:
            projs = field.projections()
            if any(abs(p) > early_stop_threshold for p in projs):
                early_stop = True
                break

    projections = field.projections()

    threshold = config.threshold
    allow_negative = config.allow_negative

    winner_candidates = []
    for i, p in enumerate(projections):
        val = abs(p) if allow_negative else p
        if val > threshold:
            winner_candidates.append((i, p))

    if allow_negative:
        winner_candidates.sort(key=lambda x: abs(x[1]), reverse=True)
    else:
        winner_candidates.sort(key=lambda x: x[1], reverse=True)

    if config.max_winners is not None:
        winner_candidates = winner_candidates[:config.max_winners]

    winners = [idx for idx, _ in winner_candidates]

    return {
        "winners": winners,
        "projections": projections,
        "cycles_used": cycles_used,
        "early_stop": early_stop
    }

def sinus_patterns(n: int, ks: list) -> list:
    patterns = []
    for k in ks:
        m = [0.0] * n
        s = 0.0
        for i in range(n):
            x = 2.0 * math.pi * float(k) * float(i) / float(n)
            m[i] = math.sin(x)
            s += m[i] * m[i]
        nrm = math.sqrt(s)
        for i in range(n):
            m[i] /= nrm
        patterns.append(m)
    return patterns
