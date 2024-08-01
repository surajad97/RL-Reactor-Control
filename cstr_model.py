###############
#  CSTR model #
###############

# Taken from http://apmonitor.com/do/index.php/Main/NonlinearControl

def cstr(x,t,u):

    # ==  Inputs == #
    Tc  = u   # Temperature of cooling jacket (K)

    # == States == #
    Ca = x[0] # Concentration of A in CSTR (mol/m^3)
    T  = x[1] # Temperature in CSTR (K)

    # == Process parameters == #
    Tf     = 350    # Feed temperature (K)
    q      = 100    # Volumetric Flowrate (m^3/sec)
    Caf    = 1      # Feed Concentration (mol/m^3)
    V      = 100    # Volume of CSTR (m^3)
    rho    = 1000   # Density of A-B Mixture (kg/m^3)
    Cp     = 0.239  # Heat capacity of A-B Mixture (J/kg-K)
    mdelH  = 5e4    # Heat of reaction for A->B (J/mol)
    EoverR = 8750   # E -Activation energy (J/mol), R -Constant = 8.31451 J/mol-K
    k0     = 7.2e10 # Pre-exponential factor (1/sec)
    UA     = 5e4    # U -Heat Transfer Coefficient (W/m^2-K) A -Area - (m^2)

    # == Equations == #
    rA     = k0*np.exp(-EoverR/T)*Ca # reaction rate
    dCadt  = q/V*(Caf - Ca) - rA     # Calculate concentration derivative
    dTdt   = q/V*(Tf - T) \
              + mdelH/(rho*Cp)*rA \
              + UA/V/rho/Cp*(Tc-T)   # Calculate temperature derivative

    # == Return xdot == #
    xdot    = np.zeros(2)
    xdot[0] = dCadt
    xdot[1] = dTdt
    return xdot

#function that performs the simulation

def simulate_CSTR(u_traj, data_simulation, repetitions):
    '''
    u_traj: Trajectory of input values
    data_simulation: Dictionary of simulation data
    repetitions: Number of simulations to perform
    '''
    # loading process operations
    Ca    = copy.deepcopy(data_simulation['Ca_dat'])
    T     = copy.deepcopy(data_simulation['T_dat'])
    x0    = copy.deepcopy(data_simulation['x0'])
    t     = copy.deepcopy(data_simulation['t'])
    noise = data_simulation['noise']
    n     = copy.deepcopy(data_simulation['n'])

    # control preparation
    u_traj = np.array(u_traj)
    u_traj = u_traj.reshape(1,n-1, order='C')
    Tc    = u_traj[0,:]

    # creating lists
    Ca_dat    = np.zeros((len(t),repetitions))
    T_dat     = np.zeros((len(t),repetitions))
    Tc_dat    = np.zeros((len(t)-1,repetitions))
    u_mag_dat = np.zeros((len(t)-1,repetitions))
    u_cha_dat = np.zeros((len(t)-2,repetitions))

    # multiple repetitions
    for rep_i in range(repetitions):
        x   = x0

        # main process simulation loop
        for i in range(len(t)-1):
            ts      = [t[i],t[i+1]]
            # integrate system
            y       = odeint(cstr,x,ts,args=(Tc[i],))
            # adding stochastic behaviour
            s       = np.random.uniform(low=-1, high=1, size=2)
            Ca[i+1] = y[-1][0] + noise*s[0]*0.1
            T[i+1]  = y[-1][1] + noise*s[1]*5
            # state update
            x[0] = Ca[i+1]
            x[1] = T[i+1]

        # data collection
        Ca_dat[:,rep_i]    = copy.deepcopy(Ca)
        T_dat[:,rep_i]     = copy.deepcopy(T)
        Tc_dat[:,rep_i]    = copy.deepcopy(Tc)

    return Ca_dat, T_dat, Tc_dat