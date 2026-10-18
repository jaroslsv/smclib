import numpy as np
import scipy.optimize as spo
from copy import deepcopy
import time
import matplotlib.pyplot as plt

def HomoTransMat(omega,dx,dy):
    return np.array([
            [np.cos(omega),     -np.sin(omega),     dx],
            [np.sin(omega),      np.cos(omega),     dy],
            [            0,                  0,      1]
        ])

def getNewPointsFromRotationTranslation(coords:np.ndarray, omega:float, dx:float, dy:float):
    """
    coords = [
        [x1, y1],
        [x2, y2],
        ...        
    ]
    """
    tmpx = coords[:,0] * np.cos(omega) - coords[:,1] * np.sin(omega) + dx
    tmpy = coords[:,0] * np.sin(omega) + coords[:,1] * np.cos(omega) + dy
    return np.array([tmpx,tmpy]).transpose()

class Circle:
    def __init__(self,pos_x:float,pos_y:float,diameter:float) -> None:
        self.x = pos_x
        self.__initial_x__ = pos_x

        self.y = pos_y
        self.__initial_y__ = pos_y

        self.d = diameter

    def reset(self):
        self.x = self.__initial_x__
        self.y = self.__initial_y__

    def printValues(self):
        print(f"pos_x = {self.x}")
        print(f"pos_y = {self.y}")
        print(f"d     = {self.d}")

    def applyRotationTranslation(self, omega:float, dx:float, dy:float):
        tmpx = self.x * np.cos(omega) - self.y * np.sin(omega) + dx
        tmpy = self.x * np.sin(omega) + self.y * np.cos(omega) + dy
        self.x = tmpx
        self.y = tmpy

    def getNewPointsFromRotationTranslation(self, omega:float, dx:float, dy:float):
        tmpx = self.x * np.cos(omega) - self.y * np.sin(omega) + dx
        tmpy = self.x * np.sin(omega) + self.y * np.cos(omega) + dy
        return np.array([tmpx,tmpy])

    @staticmethod
    def distance(C1:Circle, C2:Circle):
        return np.sqrt((C1.x-C2.x)**2 + (C1.y-C2.y)**2)

    @staticmethod
    def inner_distance(C1:Circle, C2:Circle):
        return (C1.d+C2.d)/2 - Circle.distance(C1,C2)

    @staticmethod
    def outer_distance(C1:Circle, C2:Circle):
        return (C1.d+C2.d)/2 + Circle.distance(C1,C2)


class Group:
    def __init__(self) -> None:
        self.index = 0
        self.elements = []

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index < len(self.elements):
            self.index += 1
            return self.elements[self.index-1]
        else:
            raise StopIteration

    def __getitem__(self, index):
        return self.elements[index]

    def __len__(self):
        return len(self.elements)

    def add(self,circle:Circle):
        self.elements.append(circle)

    def applyRotationTranslation(self, omega:float, dx:float, dy:float):
        for i in range(len(self.elements)):
            self.elements[i].applyRotationTranslation(omega, dx, dy)    

    def reset(self):
        for i in range(len(self.elements)):
            self.elements[i].reset()

    def getAsArray(self):
        tmp = []
        for el in self.elements:
            tmp.append([el.x,el.y])
        return np.array(tmp)

    @staticmethod
    def tschebyscheff(G1:Group,G2:Group)->float:
        if len(G1) != len(G2):
            raise ValueError("Groups must have the same number of elements for Tschebyscheff distance calculation.")        
        return max([Circle.distance(G1[i], G2[i]) for i in range(len(G1))])
   
    @staticmethod
    def minimize(G_ref, G_err):
        if len(G_ref) != len(G_err):
            raise ValueError("Groups must have the same number of elements")
        B, A = G_ref.getAsArray(), G_err.getAsArray()

        def dist(z):
            return np.linalg.norm(B - getNewPointsFromRotationTranslation(A, *z[:3]), axis=1)

        z0 = np.r_[0.0, 0.0, 0.0, dist(np.zeros(4)).max()]   # [omega, dx, dy, s]
        res = spo.minimize(lambda z: z[3], z0,
                           constraints=[{"type": "ineq", "fun": lambda z: z[3] - dist(z)}],
                           method="SLSQP", tol=1e-9)
        return res.x[:3]

    @staticmethod
    def inner_distances(G1:Group,G2:Group)->np.ndarray:
        if len(G1) != len(G2):
            raise ValueError("Groups must have the same number of elements for inner distance calculation.")        
        return np.array([Circle.inner_distance(G1[i], G2[i]) for i in range(len(G1))])

    @staticmethod
    def outer_distances(G1:Group,G2:Group)->np.ndarray:
        if len(G1) != len(G2):
            raise ValueError("Groups must have the same number of elements for outer distance calculation.")        
        return np.array([Circle.outer_distance(G1[i], G2[i]) for i in range(len(G1))])


def clearance(G1:Group,G2:Group,D:list,typeOfFit:list):
    if len(G1) != len(G2) or len(G1) != len(D) or len(G1) != len(typeOfFit):
        raise ValueError("Arguments must have the same number of elements")
    
    # typeOfFit: 0 = inner (e.g. cylinder pin), 1 = outer (e.g. cylinder bore)
    tmp = []
    for i in range(len(G1)):
        if typeOfFit[i] == 0:
            tmp.append(Circle.inner_distance(G1[i], G2[i]) - D[i])
        elif typeOfFit[i] == 1:
            tmp.append(Circle.outer_distance(G1[i], G2[i]) - D[i])
        else:
            raise ValueError("typeOfFit must be either 0 (inner) or 1 (outer)")
    
    return np.array(tmp)


def randomUniformPointsInCircle(radius:float, N:int, rng:np.random.Generator)->np.ndarray:
    angles = rng.uniform(0, 2*np.pi, N)
    radii = radius * rng.triangular(left=0, mode=1, right=1, size=N)   

    return np.array([
        radii * np.cos(angles),
        radii * np.sin(angles)
    ])

if __name__ == "__main__":    
    rng = np.random.default_rng()
    N = int(1e4)

    ########################
    ## Toleranzen
   
    TP = 0.04  # Positionstoleranz der Pins
    diameter_Pin_bores = (4.02, 4.05)  # Durchmesser der Pinbohrungen
    diameter_cyl_pins = (4-0.008, 4)  # Durchmesser der Zylinderstifte

    ########################

    LA = 15
    AA = np.deg2rad(25)     

    LB = 15
    AB = np.deg2rad(210)    

    clearance_data = [] 
    axis_distanes = []
    tschebyscheff_results = []
    lables = ["clearance Body D19 -- Magnet housing -- Head D19","clearance Pin A", "clearance Pin B"]

    start_time = time.time()
    divider = int(N/50)
    for i in range(N):
        if i % divider == 0:
            print(f"Iteration {i}/{N} ({(i/N)*100:.1f}%)", end="\r",flush=True)

        ### Body ###
        Circ_0 = Circle(0, 0, rng.uniform(19, 19+0.03))

        # Pin Bohrung A    
        rOA = np.array([LA * np.cos(AA),LA * np.sin(AA)]) + randomUniformPointsInCircle(TP/2, 1, rng).flatten()
        DA = rng.uniform(diameter_Pin_bores[0], diameter_Pin_bores[1])
        Circ_A = Circle(rOA[0], rOA[1], DA)

        # Pin Bohrung B
        rOB = np.array([LB * np.cos(AB),LB * np.sin(AB)]) + randomUniformPointsInCircle(TP/2, 1, rng).flatten()
        DB = rng.uniform(diameter_Pin_bores[0], diameter_Pin_bores[1])
        Circ_B = Circle(rOB[0], rOB[1], DB)

        # Aussendruchmesser D
        rOD =  randomUniformPointsInCircle(0.025, 1, rng).flatten()
        DD = rng.uniform(39-0.05, 39)
        Circ_D = Circle(rOD[0], rOD[1], DD)

        Body = Group()
        Body.add(Circ_0)
        Body.add(Circ_A)
        Body.add(Circ_B)
        Body.add(Circ_D)


        ### Head ###
        Circ_0_star = Circle(0, 0, rng.uniform(19, 19+0.03))

        # Pin Bohrung A
        rOA_star = np.array([LA * np.cos(AA),LA * np.sin(AA)]) + randomUniformPointsInCircle(TP/2, 1, rng).flatten()
        DA_star = rng.uniform(diameter_Pin_bores[0], diameter_Pin_bores[1])
        Circ_A_star = Circle(rOA_star[0], rOA_star[1], DA_star)

        # Pin Bohrung B
        rOB_star = np.array([LB * np.cos(AB),LB * np.sin(AB)]) + randomUniformPointsInCircle(TP/2, 1, rng).flatten()
        DB_star = rng.uniform(diameter_Pin_bores[0], diameter_Pin_bores[1])
        Circ_B_star = Circle(rOB_star[0], rOB_star[1], DB_star)

        # Aussendruchmesser D     
        rOD_star =  randomUniformPointsInCircle(0.025, 1, rng).flatten()
        DD_star = rng.uniform(39-0.05, 39)
        Circ_D_star = Circle(rOD_star[0], rOD_star[1], DD_star)

        Head = Group()
        Head.add(Circ_0_star)
        Head.add(Circ_A_star)
        Head.add(Circ_B_star)
        Head.add(Circ_D_star)

        ### Calculate Tschebyscheff distance and optimal transformation ###
        o,dx,dy = Group.minimize(Body,Head)
        Head.applyRotationTranslation(o,dx,dy)
        tschebyscheff_results.append(Group.tschebyscheff(Body, Head))

        clearances = clearance(
            Body,
            Head,
            [rng.uniform(19-0.03,19-0.01),rng.uniform(diameter_cyl_pins[0], diameter_cyl_pins[1]),rng.uniform(diameter_cyl_pins[0], diameter_cyl_pins[1]),0],
            [0,0,0,1]
        )[0:-1]

        clearance_data.append(clearances)
        axis_distanes.append(Circle.distance(Body[3],Head[3]))
        
    end_time = time.time()
    print(f"Simulation finished in {end_time-start_time:.2f} seconds.\t\t")
    print(f"{1000*(end_time-start_time)/N:.2f} ms per run\t\t")

    clearance_data = np.array(clearance_data).transpose()
    for i, entry in enumerate(clearance_data):
        counts, bins = np.histogram(entry, bins=100)

        plt.figure(time.thread_time_ns())
        plt.subplot(2,1,1)
        plt.stairs(counts, bins)
        plt.title(f"PDF of {lables[i]}")
        plt.xlabel("Clearance [mm]")
        plt.ylabel("Frequency")
        plt.grid()

        plt.subplot(2,1,2)
        plt.plot(bins[:-1], np.cumsum(counts)/np.sum(counts))
        plt.title(f"CDF of {lables[i]}")
        plt.xlabel("Clearance [mm]")
        plt.ylabel("Probability")
        plt.grid()

        plt.show(block=False)


    ### Plot axis distance distribution
    counts, bins = np.histogram(axis_distanes, bins=100)
    plt.figure(time.thread_time_ns())

    plt.subplot(2,1,1)
    plt.stairs(counts, bins)
    plt.title(f"PDF of axis distance between Body and Head")
    plt.xlabel("Deviation [mm]")
    plt.ylabel("Frequency")
    plt.grid()

    plt.subplot(2,1,2)
    plt.plot(bins[:-1], np.cumsum(counts)/np.sum(counts))
    plt.title(f"CDF of axis distance between Body and Head")
    plt.xlabel("Deviation [mm]")
    plt.ylabel("Probability")
    plt.grid()
    plt.show(block=False)    

    ### Plot Tschebyscheff distance distribution
    counts, bins = np.histogram(tschebyscheff_results, bins=100)
    plt.figure(time.thread_time_ns())

    plt.subplot(2,1,1)
    plt.stairs(counts, bins)
    plt.title(f"PDF of Tschebyscheff distances")
    plt.xlabel("Deviation [mm]")
    plt.ylabel("Frequency")
    plt.grid()

    plt.subplot(2,1,2)
    plt.plot(bins[:-1], np.cumsum(counts)/np.sum(counts))
    plt.title(f"CDF of Tschebyscheff distances")
    plt.xlabel("Deviation [mm]")
    plt.ylabel("Probability")
    plt.grid()
    plt.show(block=False)    

    print("Simulation finished.\t\t\t")

    input("Press Enter to exit...")