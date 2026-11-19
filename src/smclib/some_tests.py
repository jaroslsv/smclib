import smclib as smc
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.RandomState(0)

### TEST 1
tmp0 = smc.dist_uniform(-10,10,rng)
tmp0.distribute(100000)
counts0, bins0 = np.histogram(tmp0.distribution,bins=100)
#print((counts0,bins0))

tmp1 = smc.dist_normal(0,3,rng)
tmp1.distribute(100000)
counts1, bins1 = np.histogram(tmp1.distribution,bins=100)
#print((counts1,bins1))

plt.stairs(counts0,bins0)
plt.stairs(counts1,bins1)
plt.show()

### TEST 2
tmp2 = smc.dist_uniform_points_in_circle(10,rng, (10,-50))
tmp2.distribute(2000)
plt.plot(tmp2.distribution[0],tmp2.distribution[1],linewidth=0, marker='o',markersize=0.4)
plt.grid()
plt.show()

### TEST 3
def fcn1(a,b,c,d):
    return a+b+c

def fcn2(a,b,c,d):
    return np.array([
        a-b/c+d,
        c*d+a, 
        a+b+c+d
    ])

def fcn3(a,b,c,d):
    return np.linalg.norm(np.array([a-c,b-d]),ord=2,axis=0)

rng = np.random.RandomState(0)

sim = smc.MCSim([fcn1, fcn2, fcn3], int(2e5))
sim.addDistribution(smc.dist_uniform_points_in_circle(0.1,rng,(0,0),"Pts 1")) # a,b
sim.addDistribution(smc.dist_uniform_points_in_circle(0.1,rng,(0,5),"Pts 2")) # c,d
sim.simulate()
sim.computePDF_CDF()

pass
exit()

edges, pdf, cdf = smc.PDF_CDF.getPDF_CDF(sim.results[2])
plt.subplot(1,2,1)
plt.stairs(values=pdf, edges=edges)
plt.title("PDF")

plt.subplot(1,2,2)
plt.stairs(values=cdf, edges=edges)
plt.title("CDF")

plt.show()