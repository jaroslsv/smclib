import numpy as np
import scipy.optimize as spo
import multiprocessing as MP
import matplotlib.pyplot as plt
import os
from smclib import uniform_pts_in_circle

def transformed_r(r:np.ndarray,th:float,x:float,y:float):
    # Must have r.shape --> (n,2)
    # r should be astype(float)
    cos_th = np.cos(th)
    sin_th = np.sin(th)
    vec = np.zeros_like(r)
    vec[:, 0] = r[:, 0] * cos_th - r[:, 1] * sin_th + x
    vec[:, 1] = r[:, 0] * sin_th + r[:, 1] * cos_th + y
    return vec

def negative_clearances(p:tuple,r0:np.ndarray, r1:np.ndarray, d0, d1, d3, sign:np.ndarray):
    # cases for 'p'
    # p = +1 := 1x pin in 2x bores
    # p = -1 := 1x hull over 2x pins
    # r1.shape must be r2.shape
    # th, x, y = p  
    return -(sign * ((d0+d1)/2 - d3) - np.linalg.norm((r0-transformed_r(r1,p[0],p[1],p[2])), ord=2, axis=1))

def cons_jac(x, r0, r1):
    th = x[0]
    c, s = np.cos(th), np.sin(th)
    e = r0 - transformed_r(r1, th, x[1], x[2])
    dist = np.maximum(np.linalg.norm(e, axis=1), 1e-12)
    dT = np.stack([-r1[:, 0]*s - r1[:, 1]*c, r1[:, 0]*c - r1[:, 1]*s], axis=1)
    J = np.empty((len(r0), 4))
    J[:, 0] = np.sum(e*dT, axis=1) / dist
    J[:, 1] = e[:, 0] / dist
    J[:, 2] = e[:, 1] / dist
    J[:, 3] = 1.0
    return J

def worker2(params:list):
    # goal: maximize minimum clearance
    # x = [theta, x, y, u]
    r_vecs_0, r_vecs_1, d0, d1, d3, sign = params
    results = []
    for i in range(len(r_vecs_0)):
        args = r_vecs_0[i], r_vecs_1[i], d0[i], d1[i], d3[i], sign[i]
        mres = spo.minimize(fun = lambda x : x[3],
            x0 = (0,0,0,np.max(negative_clearances((0, 0, 0), *args))),
            constraints={
                "type":"ineq",
                "fun":lambda x : x[3] - negative_clearances((x[0], x[1], x[2]), *args),
                "jac": lambda x: cons_jac(x, args[0], args[1])
                },
            method="SLSQP",
            jac=lambda x: np.array([0., 0., 0., 1.])
            )
        results.append(mres)
    return results

class Pair:
    def __init__(self, d0:tuple, d1:tuple, d3:tuple, r0_pos:np.ndarray, r0_tol:float, r1_pos:np.ndarray, r1_tol:float, N:int, ctype:int, rng:np.random.RandomState) -> None:
        self.d0 = rng.uniform(d0[0], d0[1], N)
        self.d1 = rng.uniform(d1[0], d1[1], N)
        self.r0 = r0_pos + uniform_pts_in_circle(r0_tol,N, rng)
        self.r1 = r1_pos + uniform_pts_in_circle(r1_tol,N, rng)
        self.d3 = rng.uniform(d3[0], d3[1], N)
        self.ctype = ctype
        self.N = N

class Pattern:
    def __init__(self) -> None:
        self.pairs = []

    def addPair(self, pair:Pair):
        self.pairs.append(pair)

    def calculateParametersForMinimalClearances(self, processes=0)->list:
        if processes == 0:
            processes = os.cpu_count()

        N = self.pairs[0].N
        processes = min(processes, N)

        # Sample auf Achse 0, Pair auf Achse 1
        r0_set = np.stack([p.r0 for p in self.pairs], axis=1)   # (N,P,2)
        r1_set = np.stack([p.r1 for p in self.pairs], axis=1)   # (N,P,2)
        d0_set = np.stack([p.d0 for p in self.pairs], axis=1)   # (N,P)
        d1_set = np.stack([p.d1 for p in self.pairs], axis=1)   # (N,P)
        d3_set = np.stack([p.d3 for p in self.pairs], axis=1)   # (N,P)
        p_set  = np.tile(np.array([p.ctype for p in self.pairs], dtype=float), (N, 1))  # (N,P)

        # Chunks: obere Grenze exklusiv, keine Lücken
        bounds = np.linspace(0, N, processes + 1).astype(int)
        chunks = [
            (r0_set[a:b], r1_set[a:b], d0_set[a:b], d1_set[a:b], d3_set[a:b], p_set[a:b])
            for a, b in zip(bounds[:-1], bounds[1:])
        ]

        with MP.Pool(processes=processes) as pool:
            results = pool.map(worker2, chunks)
        return [item for sublist in results for item in sublist] 


if __name__ == "__main__":
    rng = np.random.RandomState(seed=0)
    N = 10000

    d_pin = (4-0.008,   4) # 4h6-Pin
    d_head_pin = (4.02, 4.05)
    d_body_pin = (4.02, 4.05)

    a_A = np.deg2rad(25)
    r_A = 15 * np.array([np.cos(a_A), np.sin(a_A)])

    a_B = np.deg2rad(120)
    r_B = 15 * np.array([np.cos(a_B), np.sin(a_B)])


    pat = Pattern()

    PinConnection_A = Pair(
        d0=d_head_pin, d1=d_body_pin, d3=d_pin,
        r0_pos=r_A, r0_tol=0.04,
        r1_pos=r_A, r1_tol=0.04,
        N=N, ctype=1, rng=rng
    )

    PinConnection_B = Pair(
        d0=d_head_pin, d1=d_body_pin, d3=d_pin,
        r0_pos=r_B, r0_tol=0.04,
        r1_pos=r_B, r1_tol=0.04,
        N=N, ctype=1, rng=rng
    )

    MagnetConnection = Pair(
        d0=(19.01, 19.04), d1=(19.00, 19.03), d3=(19-0.03,19.00),
        r0_pos=np.array([0,0]), r0_tol=0.04,
        r1_pos=np.array([0,0]), r1_tol=0.04,
        N=N, ctype=1, rng=rng
    )

    ClampingNutConnection = Pair(
        d0=(42, 42.1), d1=(42, 42.1), d3=(42.1,42.2),
        r0_pos=np.array([0,0]), r0_tol=0.05,
        r1_pos=np.array([0,0]), r1_tol=0.05,
        N=N, ctype=-1, rng=rng
    )

    pat.addPair(PinConnection_A)
    pat.addPair(PinConnection_B)
    pat.addPair(MagnetConnection)
    pat.addPair(ClampingNutConnection)

    res = pat.calculateParametersForMinimalClearances()

    #--- Daten visualisieren ---#

    # Minimum clearance
    counts, bins = np.histogram([-r.fun for r in res], bins=150)
    cdf = np.cumsum(counts)/np.sum(counts)

    plt.subplot(2,1,1)
    plt.title("PDF of min. clearance")
    plt.stairs(counts,bins)
    plt.xlabel("min. clearance / mm")
    plt.grid()

    plt.subplot(2,1,2)
    plt.title("CDF of min. clearance")
    plt.plot(bins[:-1], cdf)
    plt.xlabel("min. clearance / mm")
    plt.ylabel("cummulative probability")
    plt.grid()

    plt.show()

    # axes deviation Magnet bores von head and body
    r1_transformed = []
    for i in range(len(pat.pairs[2].r1)):
        r1_transformed.append(transformed_r(
            np.array([pat.pairs[2].r1[i]]),
            res[i].x[0],
            res[i].x[1],
            res[i].x[2]
            )[0])
    r1_transformed = np.array(r1_transformed)

    rd = np.linalg.norm(np.array(pat.pairs[2].r0) - r1_transformed, ord=2, axis=1)

    counts, bins = np.histogram(rd, bins=150)
    cdf = np.cumsum(counts)/np.sum(counts)

    plt.subplot(2,1,1)
    plt.title("PDF of axes deviation magnet bores")
    plt.stairs(counts,bins)
    plt.xlabel("distance / mm")
    plt.grid()

    plt.subplot(2,1,2)
    plt.title("CDF of axes deviation magnet bores")
    plt.plot(bins[:-1], cdf)
    plt.xlabel("distance / mm")
    plt.ylabel("cummulative probability")
    plt.grid()

    plt.show()