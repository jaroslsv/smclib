import smclib as MC
import numpy as np

"""
Let's assume following situation:

You need to connect two parts with cylinder pins

Part 1 and 2: have each two bores 5H7 with a distance of (20 +/- 0.05)mm
Both cylinder pins are 5h6

Will there be an interferance and if so, how large will it be and how
likely will it occur? --> here comes the smclib into play

Part 1
+--> Bore 1 : d11
+--> Bore 2 : d12
+--> distance between bores : L1

Part 2
+--> Bore 1 : d21
+--> Bore 2 : d22
+--> distance between bores : L2

Diameter dylinder pin 1 : d31
Diameter dylinder pin 2 : d32

"""

# define a function for the gap
# a positive gap means that there is no interfernce and vice versa
def gap(d11,d12,L1,
        d21,d22,L2,
        d31, d32):
    return 0.5*(d11+d21+d22+d12)-d31-d32-np.abs(L1-L2)

# Set up Monte-Carlo Simulation object
Sim = MC.MCSimulation(gap,int(1e6))

# Add parameteter in correct order!
Sim.AddParam(5, 0, +0.012)  # d11, 5H7
Sim.AddParam(5, 0, +0.012)  # d12, 5H7
Sim.AddParam(20, -0.05, +0.05) # L1, (20 +/- 0.05)

Sim.AddParam(5, 0, +0.012)  # d21, 5H7
Sim.AddParam(5, 0, +0.012)  # d22, 5H7
Sim.AddParam(20, -0.05, +0.05) # L2, (20 +/- 0.05)

Sim.AddParam(5, -0.008, 0)  # d31, 5h6
Sim.AddParam(5, -0.008, 0)  # d32, 5h6

# Start simulation and compute PDF und CDF
Sim.Simulate()
Sim.computePDF_CDF(N_bins=200)

# Plot results
Sim.Plot("My title","X-label","unit")
"""
The result will show that there might be an interference with a chance
of ca. 65% 
"""

# let's make the tolerances of L1 und L2 smaller
Sim = MC.MCSimulation(gap,int(1e6))
Sim.AddParam(5, 0, +0.012)  
Sim.AddParam(5, 0, +0.012)  
Sim.AddParam(20, -0.01, +0.01) # L1, (20 +/- 0.01)
Sim.AddParam(5, 0, +0.012)
Sim.AddParam(5, 0, +0.012)  
Sim.AddParam(20, -0.01, +0.01) # L2, (20 +/- 0.01)
Sim.AddParam(5, -0.008, 0) 
Sim.AddParam(5, -0.008, 0)
Sim.Simulate()
Sim.computePDF_CDF(N_bins=200)
Sim.Plot("My title","X-label","unit")
"""
The result will show that there might be an interference with a chance
of ca. 4% 
"""