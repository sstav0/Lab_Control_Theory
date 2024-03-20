import numpy as np

import matplotlib.pyplot as plt
from IPython.display import display, clear_output

def LL_RT(MV, Kp, TLead, TLag, Ts, PV, PVInit=0, method='EBD'):
    """
    The function "FO_RT" needs to be included in a "for or while loop".
    
    :MV: input vector
    :Kp: process gain
    :TLead: Lead time constant [s]
    :TLag: Lag time constant [s]
    :Ts: sampling period [s]
    :PV: output vector
    :PVInit: (optional: default value is 0)
    :method: discretisation method (optional: default value is 'EBD')
        EBD: Euler Backward difference
        EFD: Euler Forward difference
        TRAP: Trapezoïdal method
    
    The function "FO_RT" appends a value to the output vector "PV".
    The appended value is obtained from a recurrent equation that depends on the discretisation method.
    """    
    
    if (TLag != 0):
        K = Ts/TLag
        if len(PV) == 0:
            PV.append(PVInit)
        else: # MV[k+1] is MV[-1] and MV[k] is MV[-2]
            if method == 'EBD':
                PV.append((1/(1+K))*PV[-1] + (K*Kp/(1+K))*((1+TLead/Ts)*MV[-1] - (TLead/Ts)*MV[-2]))
            elif method == 'EFD':
                PV.append((1-K)*PV[-1] + K*Kp((TLead/Ts)*MV[-1] + (1- TLead/Ts)*MV[-2]))
            # elif method == 'TRAP':
            #     PV.append((1/(2*T+Ts))*((2*T-Ts)*PV[-1] + Kp*Ts*(MV[-1] + MV[-2])))            
            else:
                PV.append((1/(1+K))*PV[-1] + (K*Kp/(1+K))*MV[-1])
    else:
        PV.append(Kp*MV[-1])


def PID_RT(SP, PV, Man, MVMan, MVFF, Kc, Ti, Td, alpha, Ts, MVMin, MVMax, MV, MVP, MVI, MVD, E, ManFF=False, PVInit=0, methodI='EBD', methodD='EBD') :
    """
    The function "PID_RT" needs to be included in a "for or while loop".

    :SP: SetPoint vector
    :PV: Process Value vector
    :Man: Manual controller mode vector [bool]
    :MVMan: Manual value for MV vector
    :MVFF: FeedForward vector
    
    :Kc: Controller gain [float]
    :Ti: Integral time constant [s]
    :Td: Derivative time constant [s]
    :alpha: Proportionality factor for Tfd = alpha*Td where Tfd is the derivative filter time constant [s]
    :Ts: Sampling period [s]
    
    :MVMin: Minimum value for MV (for saturation of MV) [float]
    :MVMax: Maximum value for MV (for saturation of MV) [float]
    
    :MV: Manipulated Value vector
    :MVP: Proportional part of MV vector
    :MVI: Integral part of MV vector
    :MVD: Derivative part of MV vector
    :E: Control Error vector
    
    :ManFF: Activating FeedForward in manual mode (optional: default False) [bool]
    :PVInit: Initial value for PV (optional: default 0) [int]
    :methodI: Discretisation method for Integral action (optional: default 'EBD') [string]
        EBD : Euler Backward Difference
        TRAP : Trapezoids
    :methodD: Discretisation method for Derivative action (optional: default 'EBD') [string]
        EBD : Euler Backward Difference
        TRAP : Trapezoids

    The function "PID_RT" appends new values to the vectors "MV", "MVP", "MVI", "MVD", and "E".
    """
    
    # Error
    if len(PV) == 0:
        E.append(SP[-1] - PVInit)
    else: 
        E.append(SP[-1] - PV[-1])
    
    # Proportionnal action
    MVP.append(Kc*E[-1])
    
    # Integral action
    if len(MVI) == 0:
        MVI.append((Kc*Ts/Ti)*E[-1]) # Start with EBD because E[-2] does not exist
    else:
        if methodI == 'TRAP':
            MVI.append(MVI[-1] + (Kc*Ts/Ti)*(E[-1] + E[-2])/2)
        else: # EBD
            MVI.append(MVI[-1] + (Kc*Ts/Ti)*E[-1])
            
    # Derivative action
    Tfd = alpha*Td
    if len(MVD) == 0:
        MVD.append((Kc*Td/(Tfd + Ts))*E[-1]) # E[-2] = 0
    else:
        if methodD == 'TRAP':
            MVD.append((Tfd - Ts/2)/(Tfd + Ts/2)*MVD[-1] + (Kc*Td/(Tfd + Ts/2))*(E[-1] - E[-2]))
        else: # EBD
            MVD.append(Tfd/(Tfd + Ts)*MVD[-1] + (Kc*Td/(Tfd + Ts))*(E[-1] - E[-2]))

    # Integrator Reset
    if Man[-1] == True:
        if ManFF == True:
            MVI[-1] = MVMan[-1] - MVP[-1] - MVD[-1]
        else :
            MVI[-1] = MVMan[-1] - MVP[-1] - MVD[-1] - MVFF[-1]

    # Saturation Integrator Reset
    if MVP[-1] + MVI[-1] + MVD[-1] + MVFF[-1] > MVMax :
        MVI[-1] = MVMax - MVP[-1] - MVD[-1] - MVFF[-1]
    elif MVP[-1] + MVI[-1] + MVD[-1] + MVFF[-1] < MVMin :
        MVI[-1] = MVMin - MVP[-1] - MVD[-1] - MVFF[-1]

    # Resulting MV
    MV.append(MVP[-1] + MVI[-1] + MVD[-1] + MVFF[-1])