import numpy as np
import smclib as MC

### Tutorial ###
def eqn1(a,b,c,d):
    return 2*a+b-0.5*c**d

def eqn2(a,b,c):
    # nicht für Vektorisierung geeignet!
    tmp = 0
    if a > 3:
        tmp = a+b+c
    else:
        tmp = a-b-0.5*c
    return tmp

def eqn2alternative(a,b,c):
    # macht dasselbe wie eqn2, jedoch mit Vektorisierung kompatibel
    return np.where(a>3,a+b+c,a-b-0.5*c)

def main():
    Sim1 = MC.MCSimulation(eqn1, 1e5)    # Setup Simulation-object
    Sim1.AddParam(10, 0.1, 0.2)      # Add argument 'a' of function 'eqn'
    Sim1.AddParam(1,-0.02,0.05)      # Add argument 'b' of function 'eqn'
    Sim1.AddParam(4,-1,0.2)            # Add argument 'c' of function 'eqn'
    Sim1.AddParam(2,-0.1,0.1)          # Add argument 'd' of function 'eqn'
    Sim1.Simulate()                  # Run MC-simulation
    Sim1.computePDF_CDF()
    Sim1.Plot("Titel","X Achse","Einheit",2.5/100,97.5/100)

    ## Dieser Ansatz gibt Fehler aus, weil eqn2 nicht vektorisierbar
    #Sim2 = MCSimulation(eqn2, 1e6)
    #Sim2.AddParam(3,-1,1)
    #Sim2.AddParam(1,1,2)
    #Sim2.AddParam(-5,-1,4)
    #Sim2.Simulate()

    # Dieser Ansatz ist korrekt, weil eqn2alternative ist vektorisierbar
    Sim2 = MC.MCSimulation(eqn2alternative, 1e6)
    Sim2.AddParam(3,-1,1)
    Sim2.AddParam(1,1,2)
    Sim2.AddParam(-5,-1,2)
    Sim2.Simulate()
    Sim2.computePDF_CDF(200)
    Sim2.Plot("Titel","X Achse","Einheit",2.5/100,97.5/100)
    print(f"Werte kleiner als {Sim2.__getValueForP__(0.2)} haben Wahrscheinlichkeit von 20% aufzutauchen")

if __name__ =="__main__":
    main()
