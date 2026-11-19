"""
Simple Library for Monte-Carlo-Simulations

Author:         Yaroslav Kovalev
"""

import numpy as np
import matplotlib.pyplot as plt
from time import thread_time_ns

def uniform_pts_in_circle(d:float,N:int,rng:np.random.RandomState):
    if d==0:
        return np.zeros((N, 2))
        
    r = rng.triangular(left=0, mode=d/2, right=d/2,size=N)
    angles = rng.uniform(0,2*np.pi,N)
    x = r*np.cos(angles)
    y = r*np.sin(angles)
    return np.array([x,y]).transpose()

class Distribution:
    def __init__(self,rng:np.random.RandomState,name:str):
        self.distribution = None
        self.rng = rng
        self.name = name

    def distribute(self,shots:int)->np.ndarray:
        # left empty on purpose --> after inheritance
        raise NotImplementedError("Subclasses must implement this method")

class dist_uniform(Distribution):
    def __init__(self,lower_limit:float, upper_limit:float, rng:np.random.RandomState, name:str=None):
        super().__init__(rng=rng,name=name)
        if lower_limit > upper_limit:
            raise Exception("lower_limit must be smaller than upper_limit")
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit

    def distribute(self,shots:int)->None:
        self.distribution = self.rng.uniform(self.lower_limit,self.upper_limit,shots)

class dist_normal(Distribution):
    def __init__(self, mean:float, sigma:float, rng:np.random.RandomState, name:str=None):
        super().__init__(rng=rng,name=name)
        self.mean = mean
        self.sigma = sigma

    def distribute(self,shots:int)->None:
        self.distribution = self.rng.normal(self.mean, self.sigma, shots)

class dist_uniform_points_in_circle(Distribution):
    def __init__(self, diameter:float, rng:np.random.RandomState, zero=(0.0,0.0), name:str=None):
        super().__init__(rng,name=name)
        self.diameter = diameter
        self.x, self.y = None, None
        self.zero = zero

    def distribute(self, shots:int)->None:
        self.distribution = uniform_pts_in_circle(self.diameter,shots,self.rng).transpose()
        self.distribution[0,:] += self.zero[0]
        self.distribution[1,:] += self.zero[1]

class MCSim:
    def __init__(self,fcns:list, shots:int):
        self.shots = shots
        self.functions = fcns # fcn(a_1, a_2, ... , a_n)

        self.distributions = [] # inputs
        self.distributed_values = [] # outputs
        self.results = [] # returned values from functions

        self.stats_PDF = []
        self.stats_CDF = []
        self.stats_bin_edges = []

    def addDistribution(self, d:Distribution):
        self.distributions.append(d)

    def simulate(self):
        self.distributed_values.clear()
        self.results.clear()        

        # distribute and store unpacked values
        for dist in self.distributions:
            dist.distribute(self.shots)

            if len(np.shape(dist.distribution)) > 1:
                for i in range(len(np.shape(dist.distribution))):
                    self.distributed_values.append(dist.distribution[i])
            else:
                self.distributed_values.append(dist.distribution)

        # apply values to function
        for fcn in self.functions:
            self.results.append(fcn(*self.distributed_values))

    @staticmethod
    def getPDF_CDF(data:np.ndarray, bins:int=100):        
        pdf, bin_edges = np.histogram(data,bins=bins)
        cdf = np.cumsum(pdf)
        cdf = cdf / np.max(cdf)
        return bin_edges, pdf, cdf

    def computePDF_CDF(self,bins=100):
        self.stats_bin_edges.clear()
        self.stats_PDF.clear()
        self.stats_CDF.clear()

        for entry in self.results:
            print(len(entry.shape))
            if len(entry.shape) == 1:
                b,p,c = self.getPDF_CDF(entry,bins=bins)
                self.stats_bin_edges.append(b)
                self.stats_PDF.append(p)
                self.stats_CDF.append(c)
            elif len(entry.shape) == 2:
                tmp_b = []
                tmp_p = []
                tmp_c = []
                for i in range(len(entry)):
                    b,p,c = self.getPDF_CDF(entry[i],bins=bins)
                    tmp_b.append(b)
                    tmp_p.append(p)
                    tmp_c.append(c)                    
                self.stats_bin_edges.append(np.array(tmp_b))
                self.stats_PDF.append(np.array(tmp_p))
                self.stats_CDF.append(np.array(tmp_c))
            else:
                raise TypeError("Dimensions must be eiter '(n,)' or '(n,m)'")

    def plotPDF_CDF(self, indices=[]):
        plt.figure(thread_time_ns())
        plt.subplot(2,1,1)



class __parameter__:
    def __init__(self,groundValue:float,lowerTol:float,upperTol:float):
        """Add parameter to Simulation. The execution order is crucial!
        E.g. Let's assume you have a dimension of (30 +0.1/+0.02)mm then
        the correct formulation would be:
        groundValue = 30
        lowerTol = +0.02
        upperTol = +0.1

        E.g. Let's assume  (-12.5 +/-0.02)mm then:
        groundValue = -12.5
        lowerTol = +0.02
        upperTol = -0.02

        E.g. Let's assume  (5 -0.05/-0.1)mm then:
        groundValue = 5
        lowerTol = -0.1
        upperTol = -0.05

        Args:
            groundValue: groud value
            lowerTol: lowest possibe difference to ground value
            upperTol: largest possibe difference to ground value

        Raises:
            Exception: _description_
        """
        if upperTol<lowerTol:
            raise Exception("upperTol must be lower than lowerTol!")

        self.groundValue = groundValue
        self.upperTol = upperTol
        self.lowerTol = lowerTol

class MCSimulation:
    """
    self.pdf[0] --> histogram, counts
    self.pdf[1] --> histogram, bins

    self.cdf[0] --> probability 0...1 for self.cdf[1]
    self.cdf[1] --> values
    """

    def __init__(self, function, shots:int):
        """Create MCSimulation object

        Args:
            function (function): function must accecpt vectorised arguments e.g. numpy
            shots (int): number of tests
        """
        self.function = function
        self.shots = int(shots)
        self.params = []
        self.results = []
        self.pdf = []
        self.cdf = []
    
    def __getValueForP__(self, p):
        """
        p = probability 0...1

        cdf[0] = Values
        cdf[1] = 0...1
        """ 
        return np.interp(p,self.cdf[1],self.cdf[0])
        
    def AddParam(self, groundValue:float, lowerTol:float, upperTol:float):
        """Add parameter to Simulation. The execution order is crucial!
        E.g. Let's assume you have a dimension of (30 +0.1/+0.02)mm then
        the correct formulation would be:
        groundValue = 30
        lowerTol = +0.02
        upperTol = +0.1

        E.g. Let's assume  (-12.5 +/-0.02)mm then:
        groundValue = -12.5
        lowerTol = +0.02
        upperTol = -0.02

        E.g. Let's assume  (5 -0.05/-0.1)mm then:
        groundValue = 5
        lowerTol = -0.1
        upperTol = -0.05

        Args:
            groundValue (_type_): _description_
            lowerTol (_type_): _description_
            upperTol (_type_): _description_
        """
        self.params.append(__parameter__(groundValue,lowerTol, upperTol))
    
    def Simulate(self):
        paramsValuesList = []
        rng = np.random.default_rng()
        
        # Generate Random Numbers
        for p in self.params:
            paramsValuesList.append(
                rng.uniform(
                    p.groundValue + p.lowerTol,
                    p.groundValue + p.upperTol,
                    self.shots)
                )
        
        # Put generated random numbers in fcn and store result in self.results
        self.results.clear()
        paramsValuesList = np.array(paramsValuesList)        
        self.results = self.function(*paramsValuesList)
    
    def computePDF_CDF(self, N_bins=100):
        self.pdf.clear()
        self.pdf = [[],[]]

        self.cdf.clear()
        self.cdf = [[],[]]        

        self.pdf[0], self.pdf[1] = np.histogram(
            self.results,
            range=(min(self.results),max(self.results)),
            bins=N_bins
        )
        
        self.cdf[1] = np.cumsum(self.pdf[0]) / len(self.results)
        self.cdf[0] = np.linspace(
            min(self.results),
            max(self.results),
            len(self.cdf[1]))
        
    def Plot(self, diagramTitle:str, xlabel:str, unit:str, lowerP=-1.0, upperP=-1.0):
        """
        Docstring for Plot
        
        :param self: self
        :param diagramTitle: Title of diagram
        :type diagramTitle: str
        :param xlabel: Label of x-axis
        :type xlabel: str
        :param unit: Label of unit e.g. mm, MPa
        :type unit: str
        :param lowerP: show lower probability line 0.0 .. 1.0, if this parameter is lower than 0, no line will be shown. lowerP must be lower than upperP
        :param upperP: show upper probability line 0.0 .. 1.0, if this parameter is lower than 0, no line will be shown.
        """
        lowerP_Value = 0
        upperP_Value = 0
        if not(lowerP <= 0 or upperP <= 0):
            lowerP_Value = self.__getValueForP__(lowerP)
            upperP_Value = self.__getValueForP__(upperP)
        
        mean = np.mean(self.results)

        plt.subplot(2, 1, 1)
        plt.stairs(self.pdf[0], self.pdf[1], label="PDF")
        plt.plot([mean] * 2, [0, np.max(self.pdf[0])], label="Mittelwert")
        if not(lowerP <= 0 or upperP <= 0):
            plt.plot([lowerP_Value,lowerP_Value],[0, np.max(self.pdf[0])], label=f"Untere Schranke: {lowerP}")
            plt.plot([upperP_Value,upperP_Value],[0, np.max(self.pdf[0])], label=f"Obere Schranke: {upperP}")
        plt.ylabel("Anzahl Treffer")
        plt.xlabel(f"{xlabel} [{unit}]")
        plt.title(f"{diagramTitle}\nMonte-Carlo-Simulation mit {self.shots} Shots")
        plt.grid()
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(
            self.cdf[0],self.cdf[1], label="CDF"
        )
        plt.ylabel("kummulative Wahrscheinlichkeit")
        plt.xlabel(f"{xlabel} [{unit}]")
        plt.grid()
        plt.legend()
        plt.show()
