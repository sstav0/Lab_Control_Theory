import numpy as np
import sys
import subprocess

import matplotlib.pyplot as plt
from package_DBR import Process


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
    :method: discretization method (optional: default value is 'EBD')
        EBD: Euler Backward difference
        EFD: Euler Forward difference
        TRAP: Trapezoidal method
    
    The function "FO_RT" appends a value to the output vector "PV".
    The appended value is obtained from a recurrent equation that depends on the discretization method.
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
    :methodI: Discretization method for Integral action (optional: default 'EBD') [string]
        EBD : Euler Backward Difference
        TRAP : Trapezoids
    :methodD: Discretization method for Derivative action (optional: default 'EBD') [string]
        EBD : Euler Backward Difference
        TRAP : Trapezoids

    The function "PID_RT" appends new values to the vectors "MV", "MVP", "MVI", "MVD", and "E".
    """
    
    # Error
    if len(PV) == 0:
        E.append(SP[-1] - PVInit)
    else: 
        E.append(SP[-1] - PV[-1])
    
    # Proportional action
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


def IMCTuning(K, T1p, T2p=0.0, theta=0.0, gamma=0.5, model="SOPDT") :
    """
    "IMCTuning" returns the tuning parameters for a PID controller based on the Internal Model Control (IMC) method.
    
    :K: Process gain
    :T1p: First process time constant [s]
    :T2p: Second process time constant [s]
    :theta: Delay [s]
    :gamma: Factor between closed-loop time constant and open-loop time constant `T_CLP = gamma * T_OLP`
    :model: Process model (optional: default is 'SOPDT')
        FOPDT: First Order Plus Dead Time
        SOPDT: Second Order Plus Dead Time
    
    :returns: tuple with Kc, Ti, Td
    """
    
    T_CLP = gamma * T1p
    
    if model == "SOPDT":
        # SOPDT
        Kc = (1/K)*(T1p+T2p)/(T_CLP+theta)
        Ti = T1p + T2p
        Td = (T1p * T2p)/(T1p + T2p)
    else:
        # FOPDT
        Kc= (1/K)*(T1p + theta/2)/(T_CLP + theta/2)
        Ti = T1p + theta/2
        Td = (T1p * theta)/(2*T1p + theta)
        
    return Kc, Ti, Td


class Controller:
    
    def __init__(self, parameters):
        
        self.parameters = parameters
        self.parameters['Kc'] = parameters['Kc'] if 'Kc' in parameters else 1.0
        self.parameters['Ti'] = parameters['Ti'] if 'Ti' in parameters else 0.0
        self.parameters['Td'] = parameters['Td'] if 'Td' in parameters else 0.0
        self.parameters['Tfd'] = self.parameters['Tfd'] if 'Tfd' in parameters else 0.0


def Margins(P: Process, C: Controller, omega, show=True) :
    """
    "Margins" plot the Bode diagram of the loop `L(s) = P(s)*C(s)` and calculate gain & phase margins.
    
    :P: Process as defined by the class "Process".
        Use the following command to define the default process which is simply a unit gain process:
            `P = Process({})`
        
        Use the following commands for a SOPDT process:
            `P = Process({'Kp' : 1.1, 'theta' : 2.0, 'Tlag1' : 10.0, 'Tlag2' : 2.0})`
            
        Use the following commands for a unit gain Lead-lag process:
            `P = Process({'Tlag1' : 10.0, 'Tlead1' : 15.0})`
            
    :C: Controller as defined by the class "Controller".
        Use the following command to define the default controller which is simply a unit gain controller:
            `C = Controller({})`
        
        Use the following commands for a PID controller:
            `C = Controller({'Kc' : 5.0, 'Ti' : 180.0, 'Td' : 0.8, 'Tfd' : alpha * 0.8})`
            
    :omega: frequency vector of the form `omega = np.logspace(-4, 1, 10000)`
    :show: show plot (optional: default is True)
        If show == True, the Bode diagram of L(s) is shown.
        If show == False, the Bode diagram is NOT shown and the complex vector Ls, gain margin GM, phase margin PM are returned.
        
    :returns: tuple with GM, PM
    """
    s = 1j * omega
    
    # Process transfer function
    Ptheta = np.exp(-P.parameters['theta']*s)
    PGain = P.parameters['Kp']*np.ones_like(Ptheta)
    PLag1 = 1/(P.parameters['Tlag1']*s + 1)
    PLag2 = 1/(P.parameters['Tlag2']*s + 1)
    PLead1 = P.parameters['Tlead1']*s + 1
    PLead2 = P.parameters['Tlead2']*s + 1
    
    Ps = np.multiply(Ptheta,PGain)
    Ps = np.multiply(Ps,PLag1)
    Ps = np.multiply(Ps,PLag2)
    Ps = np.multiply(Ps,PLead1)
    Ps = np.multiply(Ps,PLead2)
    
    # Controller transfer function
    Cs = C.parameters['Kc'] * (1 + 1/(C.parameters['Ti']*s) + C.parameters['Td']*s/(1 + C.parameters['Tfd']*s))
    
    # Loop transfer function
    Ls = np.multiply(Ps,Cs)

    # Gain margin
    GM = 0
    ultimate_freq = 0
    phase_crossing_idx = np.argmin(np.abs(np.angle(Ls, deg=True) - -180)) # Find the nearest point with an angle of -180°
    if phase_crossing_idx > 0:
        ultimate_freq = omega[phase_crossing_idx]
        GM = 20*np.log10(1 / np.abs(Ls[phase_crossing_idx]))
        print(f"Gain margin GM = {GM:.2f} dB at {ultimate_freq:.2f} rad/s")
    else:
        print(">> Index for which arg(Ls) = -180° not found")
    
    # Phase margin
    PM = 0
    crossover_freq = 0
    gain_crossing_idx = np.argmin(np.abs(np.abs(Ls) - 1)) # Find the nearest point with a gain of 1
    if gain_crossing_idx > 0:
        crossover_freq = omega[gain_crossing_idx]
        PM = 180 + np.angle(Ls[gain_crossing_idx], deg=True)
        print(f"Phase margin PM = {PM:.2f}° at {crossover_freq:.2f} rad/s")
    else:
        print(">> Index for which |Ls| = 1 not found")
        
    
    if show == True:
        fig, (ax_gain, ax_phase) = plt.subplots(2,1)
        fig.set_figheight(10)
        fig.set_figwidth(18)
        
        # Gain Bode plot
        ax_gain.semilogx(omega,20*np.log10(np.abs(Ls)), label=r'$L(s)=P(s)C(s)$')
        ax_gain.semilogx(omega,20*np.log10(np.abs(Ps)), ':',label=r'$P(s)$')
        ax_gain.semilogx(omega,20*np.log10(np.abs(Cs)), ':',label=r'$C(s)$')
        ax_gain.axhline(0, color='black', linestyle='--')
        ax_gain.vlines(ultimate_freq, -GM, 0, color='red')
        gain_min = np.min(20*np.log10(np.abs(Ls)/5))
        gain_max = np.max(20*np.log10(np.abs(Ls)*5))
        ax_gain.set_xlim([np.min(omega), np.max(omega)])
        ax_gain.set_ylim([gain_min, gain_max])
        ax_gain.set_ylabel('Amplitude' + '\n$|L(s)|$ [dB]')
        ax_gain.set_title('Bode plot of L')
        ax_gain.legend(loc='best')
        ax_gain.grid()
        
        # Phase Bode plot
        ax_phase.semilogx(omega, (180/np.pi)*np.unwrap(np.angle(Ls)),label=r'$L(s)=P(s)C(s)$')
        ax_phase.semilogx(omega, (180/np.pi)*np.unwrap(np.angle(Ps)), ':',label=r'$P(s)$')
        ax_phase.semilogx(omega, (180/np.pi)*np.unwrap(np.angle(Cs)), ':',label=r'$C(s)$')
        ax_phase.axhline(-180, color='black', linestyle='--')
        ax_phase.vlines(crossover_freq, -180, -180 + PM, color='blue')
        ax_phase.set_xlim([np.min(omega), np.max(omega)])
        ph_min = np.min((180/np.pi)*np.unwrap(np.angle(Ls))) - 10
        ph_max = np.max((180/np.pi)*np.unwrap(np.angle(Ls))) + 10
        ax_phase.set_ylim([np.max([ph_min, -200]), ph_max])
        ax_phase.set_xlabel(r'Frequency $\omega$ [rad/s]')
        ax_phase.set_ylabel('Phase' + '\n' + r'$\angle L(s)$ [°]')
        ax_phase.legend(loc='best')
        ax_phase.grid()

    else:
        return Ls, GM, PM
    

def install_and_import(package):
    """
    Tries to import a package, if it fails, it installs the package using pip and tries to import it again.

    Parameters
    ----------
    package : str
        The name of the package to import.
    """
    try:
        __import__(package)
    except ImportError:
        print(f"{package} not found, installing using pip...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        # Try importing again after installing
        __import__(package)