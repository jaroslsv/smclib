"""
Simple Library for Monte-Carlo-Simulations

Author:         Yaroslav Kovalev
"""

import numpy as np
import matplotlib.pyplot as plt

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