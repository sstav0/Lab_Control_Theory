import numpy as np
import sys
import subprocess
import time
import tclab
import threading

import matplotlib.pyplot as plt
from package_DBR import Process, SelectPath_RT, Delay_RT, FO_RT

#----------------------------------#

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

#----------------------------------#

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

#----------------------------------#

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

#----------------------------------#

class Controller:
    
    def __init__(self, parameters):
        
        self.parameters = parameters
        self.parameters['Kc'] = parameters['Kc'] if 'Kc' in parameters else 1.0
        self.parameters['Ti'] = parameters['Ti'] if 'Ti' in parameters else 0.0
        self.parameters['Td'] = parameters['Td'] if 'Td' in parameters else 0.0
        self.parameters['Tfd'] = self.parameters['Tfd'] if 'Tfd' in parameters else 0.0

#----------------------------------#

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
   
#----------------------------------#

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
        
#----------------------------------#

#plotly imports
install_and_import('plotly')
install_and_import('plotly')
install_and_import('ipywidgets')
from plotly.subplots import make_subplots
import plotly.graph_objs as go
from ipywidgets import interactive, VBox, IntRangeSlider, IntSlider, Checkbox, FloatSlider, Label, Layout, Button

class ExperimentControl : 
    """
    This class creates a GUI for the real-time simulation or hardware operation of a process with a PID controller.
    It allows adjusting several parameters such as the setpoint, manual mode, perturbation, feedforward, controller parameters, and the simulation time step.

    Usage:
        - Create an instance of the class with the required parameters.
        - Call `createPlot` to generate the plot.
        - Interact with the simulation through the GUI.

    Parameters:
        alpha (float): Proportionality factor for Tfd = alpha * Td, where Tfd is the derivative filter time constant [s].
        gamma (float): Ratio between closed-loop and open-loop time constants `T_CLP = gamma * T_OLP`.
        PVParam (list): Process parameters [Kp, T1, T2, theta].
        DVParam (list): Disturbance parameters [Kp, T1, T2, theta].
        DV0 (float): Initial value for the disturbance.
        MV0 (float): Initial value for the manipulated variable.
        PV0 (float): Initial value for the process variable.
        MVMin (float): Minimum value for the manipulated variable.
        MVMax (float): Maximum value for the manipulated variable.
        TSim (float): Simulation time [s].
        Ts (float): Sampling time [s].

    Methods:
        initialize(): Initializes the figure.
        reinitialize(): Reinitializes arrays used to compute the PID function.
        RunExp(Exp): Core function; runs the experiment.
        Update_gamma(gamma): Updates the gamma value with the slider's value.
        stop_experiment_button_clicked(b): Stops the experiment when the stop button is clicked.
        Update_alpha(alphaP): Updates the alpha value with the slider's value.
        Update_Manual(manual_time, MVManual, Man): Updates the manual mode with the slider's value.
        Update_Perturbation(perturbation, time_perturbation): Updates the perturbation with the slider's value.
        Update_SetPoint(setpoint, time_setpoint): Updates the setpoint with the slider's value.
        Update_TimeStep(_timeStep): Updates the time step with the slider's value.
        run_experiment_threaded(): Runs the experiment in a separate thread.
        run_experiment_button_clicked(b): Runs the experiment when the run button is clicked.
        clear_graph_button_clicked(b): Clears the graph when the clear button is clicked.
        Update_FF(change): Updates the feedforward with the checkbox's value.
        Update_ManFF(change): Updates the manual feedforward with the checkbox's value.
        create_button(description, button_style, icon): Creates a button with the specified style.
        Update_slider_on_tclab_change(change): Updates the TCLab status with the checkbox's value.
        createWidgetsStyle(): Creates the style for the widgets.
        createWidgets(): Creates the widgets.
        createPlot(): Creates and returns the plot layout.
    """
    def __init__(self, alpha, gamma, PVParam, DVParam, DV0, MV0, PV0, MVMin, MVMax, TSim, Ts):
        self.alpha = alpha
        self.gamma = gamma
        self.Kp_OMV_SOPDT, self.T1_OMV_SOPDT, self.T2_OMV_SOPDT, self.theta_OMV_SOPDT = PVParam
        self.Kp_ODV_SOPDT, self.T1_ODV_SOPDT, self.T2_ODV_SOPDT, self.theta_ODV_SOPDT = DVParam
        self.DV0 = DV0
        self.MV0 = MV0
        self.PV0 = PV0
        self.MVMin = MVMin
        self.MVMax = MVMax
        
        self.Ts = Ts
        self.TSim = TSim
        self.N = int(TSim/Ts)
        self.TimeStep = Ts
        
        # Calculate IMC tuning parameters
        self.Kc, self.TI, self.TD = IMCTuning(self.Kp_OMV_SOPDT, self.T1_OMV_SOPDT, self.T2_OMV_SOPDT, self.theta_OMV_SOPDT, self.gamma)
        
        #Buffer variables
        self.KcBuffer = self.Kc
        self.TIBuffer = self.TI
        self.TDBuffer = self.TD
        self.alphaBuffer = self.alpha
        self.ManBuffer = {}
        self.MVManPathBuffer = {}
        self.FFBuffer = False
        self.SPPathBuffer = {}
        self.DVPathBuffer = {}
        self.ManFFBuffer = False
        self.TCLabStatus = False
        self.should_continue = True
    
    def initialize(self):
        self.fig = go.FigureWidget(make_subplots(rows=4, cols=1, specs = [[{}], [{}], [{}], [{}]], vertical_spacing = 0.15, row_heights=[0.1, 0.4, 0.4, 0.1], subplot_titles=("Manual Mode", "MV and Components", "PV, SP and E", "Perturbation DV")))
        self.fig.add_trace(go.Scatter(name="SP"), row=3, col=1)
        self.fig.add_trace(go.Scatter(name="PV"), row=3, col=1)
        self.fig.add_trace(go.Scatter(name="E", line=dict(dash='dash')), row=3, col=1)
        self.fig.add_trace(go.Scatter(name="MV"), row=2, col=1)
        self.fig.add_trace(go.Scatter(name="MVp", line=dict(dash='dash')), row=2, col=1)
        self.fig.add_trace(go.Scatter(name="MVi", line=dict(dash='dash')), row=2, col=1)
        self.fig.add_trace(go.Scatter(name="MVd", line=dict(dash='dash')), row=2, col=1)
        self.fig.add_trace(go.Scatter(name="Man"), row=1, col=1)
        self.fig.add_trace(go.Scatter(name="MVMan"), row=1, col=1)
        self.fig.add_trace(go.Scatter(name="DV"), row=4, col=1)
        # Update layout
        self.fig['layout'].update(height=800, width=800)
        self.fig['layout']['xaxis1'].update(title='Time (s)')
        self.fig['layout']['yaxis1'].update(title='(°C)')
        self.fig['layout']['xaxis2'].update(title='Time (s)')
        self.fig['layout']['yaxis2'].update(title='MV (%)')
        self.fig['layout']['xaxis3'].update(title='Time (s)')
        self.fig['layout']['yaxis3'].update(title='(°C)')
        self.fig['layout']['xaxis4'].update(title='Time (s)')
        self.fig['layout']['yaxis4'].update(title='DV (%)')
    
    def reinitialize(self):
        self.t = []
        self.SP = []
        self.PV = []
        self.Man = []
        self.MVMan = []
        self.DV = []
        self.MVFF = []
        self.MV = []
        self.MVp = []
        self.MVi = []
        self.MVd = []
        self.E = []
        self.PV_p = []
        self.PV_d = []
        
        self.MVFF_Delay = []
        self.MVFF_LL1 = []
        self.MV_Delay_P = []
        self.MV_FO_P = []
        self.MV_Delay_D = []
        self.MV_FO_D = []
        
    def RunExp(self, Exp):
        self.Kc = self.KcBuffer
        self.TI = self.TIBuffer
        self.TD = self.TDBuffer
        self.alpha = self.alphaBuffer
        self.ManPath = self.ManPathBuffer
        self.MVManPath = self.MVManPathBuffer
        self.FF = self.FFBuffer
        self.SPPath = self.SPPathBuffer
        self.DVPath = self.DVPathBuffer
        self.ManFF = self.ManFFBuffer

        update_frequency = 50
        data_chunk = {'t': [], 'SP': [], 'PV': [], 'E': [], 'MV': [], 'MVp': [], 'MVi': [], 'MVd': [], 'MAN': [], 'MV_MAN' : [], 'DV': []}
    
        if self.TCLabStatus:
            try :
                lab = tclab.TCLab()
            except :
                print("TCLab not connected")
                return
        
        if Exp: 
            self.reinitialize()
            
            for i in range(0, self.N): 
                
                if not self.should_continue:
                    break
                  
                self.t.append(i*self.Ts)
                
                if self.t[-1] == 0:
                    last_time = time.time()
                    
                SelectPath_RT(self.SPPath, self.t, self.SP)
                SelectPath_RT(self.ManPath, self.t, self.Man)
                SelectPath_RT(self.MVManPath, self.t, self.MVMan)
                SelectPath_RT(self.DVPath, self.t, self.DV)
                
                if self.TCLabStatus:
                    self.PV.append(lab.T1)
                    lab.Q2(self.DV[-1])
                
                # FeedForward
                Delay_RT(self.DV - self.DV0*np.ones_like(self.DV), max(self.theta_ODV_SOPDT-self.theta_OMV_SOPDT, 0), self.Ts, self.MVFF_Delay)
                LL_RT(self.MVFF_Delay, -self.Kp_ODV_SOPDT/self.Kp_OMV_SOPDT, self.T1_OMV_SOPDT, self.T1_ODV_SOPDT, self.Ts, self.MVFF_LL1)
                if self.FF == True:
                    LL_RT(self.MVFF_LL1, 1, self.T2_OMV_SOPDT, self.T2_ODV_SOPDT, self.Ts, self.MVFF)
                else:
                    LL_RT(self.MVFF_LL1, 0, self.T2_OMV_SOPDT, self.T2_ODV_SOPDT, self.Ts, self.MVFF) # Set MVFF to 0 if FF is disabled
                
                # PID
                PID_RT(self.SP, self.PV, self.Man, self.MVMan, self.MVFF, self.Kc, self.TI, self.TD, self.alpha, self.Ts, self.MVMin, self.MVMax, self.MV, self.MVp, self.MVi, self.MVd, self.E, self.ManFF, self.PV0)
                
                if self.TCLabStatus : 
                    lab.Q1(self.MV[-1])
                else :
                    # Process
                    Delay_RT(self.MV, self.theta_OMV_SOPDT, self.Ts, self.MV_Delay_P, self.MV0)
                    FO_RT(self.MV_Delay_P, self.Kp_OMV_SOPDT, self.T1_OMV_SOPDT, self.Ts, self.MV_FO_P)
                    FO_RT(self.MV_FO_P, 1, self.T2_OMV_SOPDT, self.Ts, self.PV_p)
                    
                    # Disturbance
                    Delay_RT(self.DV - self.DV0*np.ones_like(self.DV), self.theta_ODV_SOPDT, self.Ts, self.MV_Delay_D)
                    FO_RT(self.MV_Delay_D, self.Kp_ODV_SOPDT, self.T1_ODV_SOPDT, self.Ts, self.MV_FO_D)
                    FO_RT(self.MV_FO_D, 1, self.T2_ODV_SOPDT, self.Ts, self.PV_d)
                    
                    self.PV.append(self.PV_p[-1] + self.PV_d[-1] + self.PV0 - self.Kp_OMV_SOPDT*self.MV0)
                
                data_chunk['t'].append(self.t[-1])
                data_chunk['SP'].append(self.SP[-1])
                data_chunk['PV'].append(self.PV[-1])
                data_chunk['E'].append(self.E[-1])
                data_chunk['MV'].append(self.MV[-1])
                data_chunk['MVp'].append(self.MVp[-1])
                data_chunk['MVi'].append(self.MVi[-1])
                data_chunk['MVd'].append(self.MVd[-1])
                data_chunk['MAN'].append(self.Man[-1])
                data_chunk['MV_MAN'].append(self.MVMan[-1])
                data_chunk['DV'].append(self.DV[-1])
                

                if i % update_frequency == 0 and self.TimeStep == 0 :
                    if i == 0:
                        self.fig.data[0].x, self.fig.data[0].y = self.t, self.SP
                        self.fig.data[1].x, self.fig.data[1].y = self.t, self.PV
                        self.fig.data[2].x, self.fig.data[2].y = self.t, self.E
                        self.fig.data[3].x, self.fig.data[3].y = self.t, self.MV
                        self.fig.data[4].x, self.fig.data[4].y = self.t, self.MVp
                        self.fig.data[5].x, self.fig.data[5].y = self.t, self.MVi
                        self.fig.data[6].x, self.fig.data[6].y = self.t, self.MVd
                        self.fig.data[7].x, self.fig.data[7].y = self.t, self.Man
                        self.fig.data[8].x, self.fig.data[8].y = self.t, self.MVMan
                        self.fig.data[9].x, self.fig.data[9].y = self.t, self.DV    
                    else:
                        with self.fig.batch_update():
                            for j, key in enumerate(data_chunk.keys()):
                                if key != 't':  # Prevent trying to extend x-axis data onto itself
                                    self.fig.data[j-1]['x'] = self.fig.data[j-1]['x'] + tuple(data_chunk['t'])
                                    self.fig.data[j-1]['y'] = self.fig.data[j-1]['y'] + tuple(data_chunk[key])
                    #Reset data_chunk for the next batch of updates
                    data_chunk = {key: [] for key in data_chunk}
                    
                elif self.TimeStep > 0: 
                    # wait to the next loop
                    elapsed = time.time() - last_time
                    time.sleep(max(0, self.TimeStep - elapsed))
                    last_time = time.time()
                    
                    with self.fig.batch_update():                
                        self.fig.data[0].x, self.fig.data[0].y = self.t, self.SP
                        self.fig.data[1].x, self.fig.data[1].y = self.t, self.PV
                        self.fig.data[2].x, self.fig.data[2].y = self.t, self.E
                        self.fig.data[3].x, self.fig.data[3].y = self.t, self.MV
                        self.fig.data[4].x, self.fig.data[4].y = self.t, self.MVp
                        self.fig.data[5].x, self.fig.data[5].y = self.t, self.MVi
                        self.fig.data[6].x, self.fig.data[6].y = self.t, self.MVd
                        self.fig.data[7].x, self.fig.data[7].y = self.t, self.Man
                        self.fig.data[8].x, self.fig.data[8].y = self.t, self.MVMan
                        self.fig.data[9].x, self.fig.data[9].y = self.t, self.DV  
                
            if self.TCLabStatus:
                lab.close()
                
    def Update_gamma(self, gamma):
        self.KcBuffer, self.TIBuffer, self.TDBuffer = IMCTuning(self.Kp_OMV_SOPDT, self.T1_OMV_SOPDT, self.T2_OMV_SOPDT, self.theta_OMV_SOPDT, gamma, model="SOPDT")
    
    def stop_experiment_button_clicked(self, b):
        self.should_continue = False  # Set the flag to False to stop the loop

    def Update_alpha(self, alphaP):
        self.alphaBuffer = alphaP
        
    def Update_Manual(self, manual_time, MVManual, Man):        
        self.ManPathBuffer = {0: False, Man: True}
        self.MVManPathBuffer = {0: 0, manual_time[0]: MVManual[0], manual_time[1]: MVManual[1]}


    def Update_Perturbation(self, perturbation, time_perturbation):
       self. DVPathBuffer = {0: self.DV0, time_perturbation[0]: perturbation[0], time_perturbation[1]: perturbation[1]}
        
    def Update_SetPoint(self, setpoint, time_setpoint):
        self.SPPathBuffer = {0: self.PV0, time_setpoint[0]: setpoint[0], time_setpoint[1]: setpoint[1]}

    def Update_TimeStep(self, _timeStep):
        self.TimeStep = _timeStep
        
    def run_experiment_threaded(self):
        # Wrapper function to run the experiment in a separate thread
        self.RunExp(Exp=True)

    def run_experiment_button_clicked(self, b):
        self.should_continue = True  # Ensure the flag is reset to True when starting
        if not self.fig.data:
            self.initialize()
            self.reinitialize()
        
        # Start the experiment in a separate thread to keep the UI responsive
        experiment_thread = threading.Thread(target=self.run_experiment_threaded)
        experiment_thread.start()


    def stop_experiment_button_clicked(self, b):
        self.should_continue = False  # Set the flag to False to stop the loop
        
    def clear_graph_button_clicked(self, b):
        # Clear the graph by resetting its data
        self.fig.data = []
        
    def Update_FF(self, change):
        self.FFBuffer = change.new
        
    def Update_ManFF(self, change):
        self.ManFFBuffer = change.new
        
    def create_button(self, description, button_style, icon):
        return Button(
            description=description,
            button_style=button_style, 
            tooltip=f'Click to {description.lower()}',
            icon=icon
        )
    
    #TCLab checkbox update function 
    def Update_slider_on_tclab_change(self, change):
        print("ok")
        self.TCLabStatus = change.new
        if change['new']:  # If TCLAB is now True
            self.simulationTimePerLoopSlider.min = 0.5
            self.simulationTimePerLoopSlider.value = max(self.simulationTimePerLoopSlider.value, 0.5)
        else:  # If TCLAB is False
            self.simulationTimePerLoopSlider.min = 0
            # Here, you might want to adjust the value only if it's currently below the new min
            if self.simulationTimePerLoopSlider.value < 0.1:
                self.simulationTimePerLoopSlider.value = 0
                
    def createWidgetsStyle(self):
        # Define common styles for sliders and buttons
        self.slider_layout = Layout(width='500px')
        self.slider_style = {'description_width': 'initial'}

        self.button_style_mappings = {
            'run': ('Run Experiment', 'info', 'play'),
            'stop': ('Stop Experiment', 'danger', 'stop'),
            'clear': ('Clear Graph', 'warning', 'eraser'),
        }
    
    def createWidgets(self):
        self.createWidgetsStyle()
        # Creating buttons with the defined function
        self.run_exp_button = self.create_button(*self.button_style_mappings['run'])
        self.stop_exp_button = self.create_button(*self.button_style_mappings['stop'])
        self.clear_graph_button = self.create_button(*self.button_style_mappings['clear'])

        self.run_exp_button.on_click(self.run_experiment_button_clicked)
        self.stop_exp_button.on_click(self.stop_experiment_button_clicked)
        self.clear_graph_button.on_click(self.clear_graph_button_clicked)

        # Create sliders
        self.manualTimeIntervalSlider = IntRangeSlider(min=0, max=self.TSim, step=1, value=[0, 500], description="Manual Time Interval", style=self.slider_style, layout=self.slider_layout)
        self.manualActivationTimeSlider = IntSlider(min=0, max=self.TSim, step=1, value=500, description="Manual Activation Time (3000 to desactivate)", style=self.slider_style, layout=self.slider_layout)
        self.manualControlValueSlider = IntRangeSlider(min=0, max=100, step=1, value=[self.MV0+15, self.MV0+15], description="Manual Control Value", style=self.slider_style, layout=self.slider_layout)
        self.perturbationValueSlider = IntRangeSlider(min=0, max=100, step=1, value=[self.DV0, self.DV0+10], description="Perturbation Value", style=self.slider_style, layout=self.slider_layout)
        self.perturbationTimeIntervalSlider = IntRangeSlider(min=0, max=self.TSim, step=10, value=[0, 1600], description="Perturbation Time Interval", style=self.slider_style, layout=self.slider_layout)
        self.setPointValueSlider = IntRangeSlider(min=0, max=100, step=1, value=[self.PV0+5, self.PV0+10], description="Set Point Value", style=self.slider_style, layout=self.slider_layout)
        self.setPointTimeIntervalSlider = IntRangeSlider(min=0, max=self.TSim, step=10, value=[0, 1000], description="Set Point Time Interval", style=self.slider_style, layout=self.slider_layout)
        self.gammaAdjustmentSlider = FloatSlider(min=0.2, max=0.9, step=0.02, value=0.5, description="Gamma Adjustment", style=self.slider_style, layout=self.slider_layout)
        self.alphaAdjustmentSlider = FloatSlider(min=0.2, max=0.9, step=0.02, value=0.7, description="Alpha Adjustment", style=self.slider_style, layout=self.slider_layout)
        
        # Create the slider with a conditional minimum value
        self.simulationTimePerLoopSlider = FloatSlider(
            min=0.1 if self.TCLabStatus else 0,  # Use the conditional min_value here
            max=4, 
            step=0.1, 
            value=max(0.1, 0) if self.TCLabStatus else 0,  # Ensure the initial value is also adjusted
            description="Simulation Time Per Loop", 
            style=self.slider_style, 
            layout=self.slider_layout
        )
        
        # Create interactive sliders
        self.GammaSliderInteractive = interactive(self.Update_gamma, gamma=self.gammaAdjustmentSlider)
        self.AlphaSliderInteractive = interactive(self.Update_alpha, alphaP=self.alphaAdjustmentSlider)
        self.ManualInteractive = interactive(self.Update_Manual, manual_time=self.manualTimeIntervalSlider, MVManual=self.manualControlValueSlider, Man=self.manualActivationTimeSlider)
        self.PerturbationInteractive = interactive(self.Update_Perturbation, perturbation=self.perturbationValueSlider, time_perturbation=self.perturbationTimeIntervalSlider)
        self.SetPointInteractive = interactive(self.Update_SetPoint, setpoint=self.setPointValueSlider, time_setpoint=self.setPointTimeIntervalSlider)
        self.TimeStepInteractive = interactive(self.Update_TimeStep, _timeStep=self.simulationTimePerLoopSlider)
        
        # Create checkboxes
        self.FFCheckBox = Checkbox(value=False, description='FeedForward')
        self.ManFFCheckBox = Checkbox(value=False, description='Manual FeedForward')
        self.TCLabCheckBox = Checkbox(value=False, description='TCLab')
        self.FFCheckBox.observe(self.Update_FF, names='value')
        self.ManFFCheckBox.observe(self.Update_ManFF, names='value')
        self.TCLabCheckBox.observe(self.Update_slider_on_tclab_change, names='value')
        
        
    def createPlot(self):
        self.initialize()
        self.createWidgets()
        # Adding labels for sections
        self.pid_parameters_title = Label(value='**PID Parameters**', layout={'width': '500px'})
        self.pid_parameters_title.style.font_weight = 'bold'
        self.pid_parameters_title.layout.justify_content = 'center'

        # Adding labels for sections
        self.simulation_title = Label(value='**Simulation**', layout={'width': '500px'})
        self.simulation_title.style.font_weight = 'bold'
        self.simulation_title.layout.justify_content = 'center'
        
        return VBox([self.fig,
        self.pid_parameters_title,
        self.GammaSliderInteractive,
        self.AlphaSliderInteractive,
        self.ManualInteractive,
        self.ManFFCheckBox,
        self.PerturbationInteractive,
        self.FFCheckBox,
        self.SetPointInteractive,
        self.simulation_title,
        self.TimeStepInteractive,
        self.TCLabCheckBox,
        self.run_exp_button,
        self.stop_exp_button,
        self.clear_graph_button
        ], layout=Layout(align_items='center'))
        