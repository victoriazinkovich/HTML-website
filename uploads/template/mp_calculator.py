import pandas as pd
import numpy as np
from scipy import optimize
import random
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import csv
import time
import integrator
import atm.core
import atm.jobs as jr
import re
import seaborn as sns


class LammpsCalculations:
    def solid(self, Tequil: float):
        import random
        seed = random.randint(1,2**31-1)
        return f"""\

#INITIALIZATION

units           metal
atom_style      atomic
boundary        p p p

#SYSTEM DEFINITION

lattice fcc 4.046 orient x 1 0 0 orient y 0 1 0 orient z 0 0 1

#lattice diamond 5.615795

region          simbox block 0 8 0 8 0 16
create_box      1 simbox
create_atoms    1 region simbox

#set atom * type/fraction 2 1.0 777
#mass  1  28.085   # Si
#mass  2  72.630   # Ge

mass            1 26.982

#FORCE FIELD

pair_style     eam/fs
pair_coeff     * * Al_mm.eam.fs Al

#pair_style      mlip mlip.ini
#pair_coeff      * *

#RUN MOLECULAR DYNAMICS

timestep        0.001 # 1 fs

fix             1 all npt temp {Tequil} {Tequil} 0.1 iso 0.0 0.0 0.1

velocity        all create {Tequil} {seed} rot yes dist gaussian


variable        peatom equal pe/atoms
variable        volatom equal vol/atoms
variable        myTemp equal temp


thermo_style    custom step temp etotal v_peatom v_volatom
thermo 1000

run             10000

fix ave_pe all ave/time 1000 5 15000 v_peatom

thermo_style    custom step temp etotal v_peatom v_volatom
thermo 1000

variable p equal "f_ave_pe" 

run             5000

print    "MYOUTCOME: $p"
"""
    
    def liquid(self, Tliq: float):
        import random
        seed = random.randint(1,2**31-1)
        return f"""\
    
#INITIALIZATION

units           metal
atom_style      atomic
boundary        p p p

#SYSTEM DEFINITION

lattice fcc 4.046 orient x 1 0 0 orient y 0 1 0 orient z 0 0 1

#lattice diamond 5.615795

region          simbox block 0 8 0 8 0 16
create_box      1 simbox
create_atoms    1 region simbox

#set atom * type/fraction 2 1.0 777
#mass  1  28.085   # Si
#mass  2  72.630   # Ge

mass            1 26.982

#FORCE FIELD

pair_style     eam/fs
pair_coeff     * * Al_mm.eam.fs Al

#pair_style      mlip mlip.ini
#pair_coeff      * *

#RUN MOLECULAR DYNAMICS

variable        Tequil index {Tliq * 1.5}

timestep        0.001 # 1 fs

fix             1 all npt temp ${{Tequil}} ${{Tequil}} 0.1 iso 0.0 0.0 0.1

velocity        all create ${{Tequil}} {seed} rot yes dist gaussian


variable        peatom equal pe/atoms
variable        volatom equal vol/atoms
variable        myTemp equal temp


thermo_style    custom step temp etotal v_peatom v_volatom
thermo 1000

run             15000

fix             1 all npt temp {Tliq} {Tliq} 0.1 aniso 0.0 0.0 0.1


thermo_style    custom step temp etotal v_peatom v_volatom
thermo 1000

run             20000

fix ave_pe all ave/time 1000 10 45000 v_peatom

thermo_style    custom step temp etotal v_peatom v_volatom
thermo 1000

variable p equal "f_ave_pe" 

run             10000

print    "MYOUTCOME: $p"
"""
    
    def error(self, w, Temp, points):
        sum = 0
        for i in range(len(points)):
            sum += (points[i] - (w[1] + w[0]*Temp[i]))**2
        return sum
    
    def solid_line(self, T_melt):
        i = 0
        Temp_solid = np.arange(T_melt*0.8,T_melt*0.6,-T_melt*0.1)
        points_solid = np.zeros(len(Temp_solid))
        
        with open('Al_mm.eam.fs', 'r') as f:
            pot = f.read()
            
        job_id_array = [jr.submit_job(jr.job.GenericLammps(self.solid(T), extra_in_files = {'Al_mm.eam.fs':pot}), metadata = {'needs_dir': True, 'n_cores_min': 10, 'n_cores_max': 128}) for T in Temp_solid]
        for job_id in job_id_array:
            jr.wait_for_job(job_id)
            result = str(jr.get_job_result(job_id))
            match = re.search('MYOUTCOME: (-\d*.\d*)', result, re.MULTILINE)
            point = float(match.group(1)) if match is not None else None
            points_solid[i] = point
            #print("Job", job_id, "done", point, i)
            i += 1
        print('Temperatures for solid line:', Temp_solid)
        print('Potential energies of solid phase:', points_solid)
        bnds = ((-1, 1), (-10, 10))
        x0 = [0,0]
        res_solid = optimize.minimize(self.error, x0, args=(Temp_solid, points_solid), method='L-BFGS-B', bounds=bnds)
        return res_solid.x[0], res_solid.x[1]
    
    def liquid_line(self, T_melt):
        i = 0
        Temp_liquid = np.arange(T_melt*1.2, T_melt*1.4, T_melt*0.1)
        points_liquid = np.zeros(len(Temp_liquid))
        
        with open('Al_mm.eam.fs', 'r') as f:
            pot = f.read()
            
        job_id_array = [jr.submit_job(jr.job.GenericLammps(self.liquid(T), extra_in_files = {'Al_mm.eam.fs':pot}), metadata = {'needs_dir': True, 'n_cores_min': 10, 'n_cores_max': 128}) for T in Temp_liquid]
        for job_id in job_id_array:
            jr.wait_for_job(job_id)
            result = str(jr.get_job_result(job_id))
            match = re.search('MYOUTCOME: (-\d*.\d*)', result, re.MULTILINE)
            point = float(match.group(1)) if match is not None else None
            points_liquid[i] = point
            i += 1
            #print("Job", job_id, "done", point, i)
        print('Temperatures for liquid line:', Temp_liquid)
        print('Potential energies of liquid phase:', points_liquid)
        bnds = ((-1, 1), (-10, 10))
        x0 = [0,0]
        res_liquid = optimize.minimize(self.error, x0, args=(Temp_liquid, points_liquid), method='L-BFGS-B', bounds=bnds)
        return res_liquid.x[0], res_liquid.x[1]
    
    def my_lammps_script(self, Tequil: float, L: int, ksol: float, bsol: float, kliq: float, bliq: float):
        import random
        seed = random.randint(1,2**31-1)
        return f"""\

units           metal
atom_style      atomic
boundary        p p p

#SYSTEM DEFINITION SOLID+LIQUID

lattice fcc 4.046 orient x 1 0 0 orient y 0 1 0 orient z 0 0 1

#lattice diamond 5.615795

region          solid block 0 {L} 0 {L} 0 {L-0.25}
region          liquid block 0 {L} 0 {L} {L-0.25} {2*L}
region          sol_liq union 2 solid liquid

create_box      1 sol_liq
create_atoms    1 region sol_liq

group           solid region solid
group           liquid region liquid
group           sol_liq region sol_liq

#set atom * type/fraction 2 1.0 777
#mass  1  28.085   # Si
#mass  2  72.630   # Ge

mass            1 26.982 #Al

#FORCE FIELD

pair_style      eam/fs
pair_coeff      * * Al_mm.eam.fs Al

#pair_style      mlip mlip.ini
#pair_coeff      * *

#RUN MOLECULAR DYNAMICS

variable        Tliq equal {1.5*Tequil}

variable        ensol equal {ksol}*{Tequil}+{bsol}       #eam
variable        enliq equal {kliq}*{Tequil}+{bliq}       #eam

variable        ensolnew equal "(v_ensol*7 + v_enliq)/8"
variable        enliqnew equal "(v_enliq*7 + v_ensol)/8"

#variable       seed index 987654321

velocity        all create {Tequil} {seed} rot yes dist gaussian

variable        peatom equal pe/atoms
variable        volatom equal vol/atoms
variable        myStep equal step
variable        myTemp equal temp

#SIMULATION PARAMETERS 1 SOLID+LIQUID Tm iso

timestep        0.001 # 1 fs

fix             1 sol_liq npt temp {Tequil} {Tequil} 0.1 iso 0.0 0.0 0.1

thermo_style    custom step temp etotal v_peatom v_volatom
thermo 1000

run             10000

#SIMULATION PARAMETERS 2 LIQUID Tm*2 iso

fix             1 liquid npt temp ${{Tliq}} ${{Tliq}} 0.1 iso 0.0 0.0 0.1

thermo_style    custom step temp etotal v_peatom v_volatom
thermo 1000

run             10000


#SIMULATION PARAMETERS 3 SOLID+LIQUID Tm aniso

fix             1 sol_liq npt temp {Tequil} {Tequil} 0.1 aniso 0.0 0.0 0.1

thermo_style    custom step temp etotal v_peatom v_volatom
thermo 1000

run 5000 #balancing run

label loop
variable i loop 1000
run 1000
if "(${{peatom}} < ${{ensolnew}})" then "print 'MYOUTCOME: solid'" &
   "jump SELF break" &
elif "(${{peatom}} > ${{enliqnew}})" &
     "print 'MYOUTCOME: liquid'" &
     "jump SELF break"
next i
jump SELF loop
label break
print "ALL DONE"
"""
    
    def data_append(self, T, L, ksol, bsol, kliq, bliq):
        if L <= 3:
            k = 10
        elif L == 4:
            k = 5
        elif L == 5:
            k = 2
        else:
            k = 1
        sol = 0
        liq = 0
        
        with open('Al_mm.eam.fs', 'r') as f:
            pot = f.read()
            
        job_id_array = [jr.submit_job(jr.job.GenericLammps(self.my_lammps_script(T, L, ksol, bsol, kliq, bliq), extra_in_files = {'Al_mm.eam.fs':pot}), metadata = {'needs_dir': True, 'n_cores_min': 10, 'n_cores_max': 128}) for i in range(k)]
        for job_id in job_id_array:
            jr.wait_for_job(job_id)
            result = str(jr.get_job_result(job_id))
            match = re.search('MYOUTCOME: (\w*)', result, re.MULTILINE)
            phase = match.group(1) if match is not None else None
            if phase == "solid":
                sol += 1
            else:
                liq += 1
        return L, T, sol, liq
    
    def temp_sl(self, L , T, sol, liq, my_dict):
        if L not in my_dict:
            my_dict.update({L: {T: [sol, liq]}})

        elif L in my_dict:
            if T not in my_dict[L]:
                my_dict[L].update({T: [sol, liq]})
            elif T in my_dict[L]:
                my_dict[L][T][0] += sol
                my_dict[L][T][1] += liq
        return my_dict
    
    
class GaussianProcess():
    #kernel
    def k_gauss(self, N_train, N_test, thN, sig_f):
        k = np.zeros((len(N_train),len(N_test)))
        for i in range(len(N_train)):
            for j in range(len(N_test)):
                k[i][j] = sig_f**2*np.exp(-(1/N_train[i] - 1/N_test[j])**2*thN**2/2)
        return k
    
    #kernel for test points
    def k_test_gauss(self, N_train, N_test, thN, sig_f):
        k_test = np.zeros((len(N_train),1))
        for i in range(len(N_train)):
            k_test[i] = sig_f**2*np.exp(-(1/N_train[i] - 1/N_test)**2*thN**2/2)
        return k_test
    
    #log likelihood
    def log_p_gauss(self, params, N_train, T_train, var_temp):
        thN, sig_f = params
        A = self.k_gauss(N_train,N_train,thN,sig_f) + var_temp*np.identity(len(N_train))
        A_inv = np.linalg.inv(A)
        log_p = ((T_train.dot(A_inv)).dot(T_train.reshape(-1,1)) + np.log(np.linalg.det(A)) + ((thN-30)/30)**2 + ((sig_f-3000)/3000)**2 + len(N_train)*np.log(2*np.pi))/2
        return log_p
    
    #meaning value of GP regression
    def f_mean(self, N_train, T_train, N_test, var_temp, thN, sig_f):
        A = self.k_gauss(N_train,N_train,thN, sig_f) + var_temp*np.identity(len(N_train))
        A_inv = np.linalg.inv(A)
        f_star = self.k_gauss(N_test,N_train,thN, sig_f).dot(A_inv).dot(T_train.reshape(-1,1))

        return f_star
    
    #variance of the value
    def var(self, N_train, N_test, var_temp, thN, sig_f):
        Var_f_star = np.zeros((len(N_test), 1))
        A = self.k_gauss(N_train,N_train,thN, sig_f) + var_temp*np.identity(len(N_train))
        A_inv = np.linalg.inv(A)
        for i in range(len(N_test)):
            Var_f_star[i] = sig_f**2*np.exp(-(1/N_test[i] - 1/N_test[i])**2*thN**2/2) - (self.k_test_gauss(N_train,N_test[i],thN,sig_f).T).dot(A_inv).dot(self.k_test_gauss(N_train,N_test[i],thN,sig_f))
        return Var_f_star
    
    #derivation of variance
    def dV(self, N_train, N_test, var_temp, thN, sig_f):
        B = self.k_gauss(N_train,N_train,thN, sig_f) + var_temp*np.identity(len(N_train))
        B_inv = np.linalg.inv(B)
        dV = np.zeros((len(N_train), len(N_train)))
        for i in range(len(N_train)):
            a_ii = np.zeros((len(N_train)))
            a_ii[i] = 1
            dV[i][i] = -(self.k_test_gauss(N_train,N_test,thN,sig_f).T).dot(B_inv).dot(var_temp*np.identity(len(N_train))).dot(a_ii*np.identity(len(N_train))).dot(var_temp*np.identity(len(N_train))).dot(B_inv).dot(self.k_test_gauss(N_train,N_test,thN,sig_f))
        return dV
    
    #!!!!!!!!!!
    #points for which L that we need to make MD
    def add_L(self, L_array, N_array, L_train, N_train, N_test, var_temp, s_0, thN, sig_f):
        var = np.full(len(L_array), 1e+6)
        sig = 1.5*s_0/N_array
        for i in range(len(L_train)):
            for j in range(len(L_array)):
                if L_array[j] == L_train[i]:
                    var[j] = var_temp[i]
                    sig[j] = s_0/N_train[i]

        v = -1*self.dV(N_array, N_test, var, thN, sig_f)
        dVar_defforts = np.zeros((len(N_array)))
        #!!!!!!!!!!!
        for j in range(len(N_array)):
            #if L_array[j] not in L_train and L_array[j] <= 6:
            #if L_array[j] != 3 and L_array[j] < 5:
            dVar_defforts[j] = v[j][j]/(L_array[j]**7*sig[j]**2)
        print('L_array:',L_array)
        print('dVar_defforts', dVar_defforts)
        return L_array[np.argmax(dVar_defforts)].round(0), np.argmax(dVar_defforts), N_array[np.argmax(dVar_defforts)].round(0)
    
    #temperature from T_melt - 1.6*sigma < T_melt < T_melt - 0.6*sigma interval that we need to add for L
    def T_low(self, T_melt, sigma):
        k = 13
        T_low = 2**k
        while True:
            if T_low > T_melt-0.6*sigma:
                T_low -= 2**(k-1)
                k -= 1
            elif T_low < T_melt-1.6*sigma:
                T_low += 2**(k-1)
                k -= 1
            else:
                break
        return T_low

    def T_high(self, T_melt, sigma):
        k = 13
        T_high = 2**k
        while True:
            if T_high > T_melt+1.6*sigma:
                T_high -= 2**(k-1)
                k -= 1
            elif T_high < T_melt+0.6*sigma:
                T_high += 2**(k-1)
                k -= 1
            else:
                break
        return T_high
    
    def sig0(self, s_0, N_train, L_train, bayes):
        res = 0
        for i in range(len(N_train)):
            result = bayes.T_melt(L_train[i], float(s_0/N_train[i]), 1e-5)
            res += result[0]
        return -res
    
        
class MainClass:
    def __init__(self, Tmelt, structure):
        self.L_array = np.arange(13)
        self.Tmelt = Tmelt
        self.structure = structure
        if structure == 'fcc':
            self.N_array = self.L_array**3*8
        elif structure == 'diamond':
            self.N_array = self.L_array**3*16
        elif structure == 'bcc':
            self.N_array = self.L_array**3*4
        elif structure == 'sc':
            self.N_array = self.L_array**3
        else:
            print('I do not know this :(')
            
        while self.N_array[0] < 100:
            self.N_array = np.delete(self.N_array, 0)
            self.L_array = np.delete(self.L_array, 0)  

    def WriteResults(self, status:str):
        if hasattr(self,'bsol') and hasattr(self,'ksol') and hasattr(self,'bliq') and hasattr(self,'kliq'):
            solid_liquid_line = f'''\
<h2><big>Potential energies of solid and liquid</big></h2>
<p><img src="lines.png" alt="solid/liquid potential energy graphs">
</p>
'''
        else:
            solid_liquid_line = ''
            
        #qwe
        if hasattr(self,'T_needed') and hasattr(self,'var_needed'):
            result_at_inf = f'''\
<p><b><big><big>Melting point:</b> {self.T_needed[0][0].round(1)} &pm; {self.var_needed[0][0].round(1)}</big></big></p>
'''
            gp_str = f'''\
<h2><big>Melting point convergence graph</big></h2>
<p><img src="GP.png" alt="Melting point convergence graph">
</p>
'''
        else:
            result_at_inf = ''
            gp_str = ''
            
        html_str = f'''\
<!DOCTYPE html>
<html>
<title>Melting point for Al</title>
<body>

<h1><big>Melting point for Al EAM</big></h1>
{result_at_inf}
<p><b><big><big>Status:</b> {status}</big></big></p>

{gp_str}

{solid_liquid_line}

</body>
</html>
'''
        with open('templates/index.html','w') as f:
            f.write(html_str)
    
    
    def Lines(self, LAMMPS, Tmelt=None):
        if Tmelt is None:
            Tmelt = self.Tmelt
        
        self.WriteResults(status = "Solid line is being calculated")
        ksol, bsol = LAMMPS.solid_line(Tmelt)
        print('ksol*T + bsol:', 'ksol = ', ksol, ',' , 'bsol = ', bsol)
        
        self.WriteResults(status = "Liquid line is being calculated")
        kliq, bliq = LAMMPS.liquid_line(Tmelt)
        print('kliq*T + bliq:', 'kliq = ', kliq, ',' , 'bliq = ', bliq)
        
        self.ksol, self.bsol, self.kliq, self.bliq = ksol, bsol, kliq, bliq
        
        plt.figure(figsize=(8, 5))
        x = np.linspace(0.6*Tmelt, 1.4*Tmelt, 1000)
        y = ksol*x+bsol
        t = kliq*x+bliq
        plt.plot(x,y, label='Potential energy of solid phase', color='royalblue')
        plt.plot(x,t, label='Potential energy of liquid phase', color='indianred')
        plt.legend()
        plt.grid()
        plt.xlabel('Temperature, K')
        plt.ylabel('Potential energy, eV')
        plt.title('Al EAM')
        plt.savefig('lines.png', facecolor='white',dpi=200)
        plt.show()
        
        return ksol, bsol, kliq, bliq

    def FirstAdd(self, GP, LAMMPS, melt, name, sig_start, ksol, bsol, kliq, bliq, Tmelt=None):
        if Tmelt is None:
            Tmelt = self.Tmelt
        Dict_new = {}
        
        T1_start = GP.T_low(Tmelt, sig_start)
        print('Temperatures of MD for the first L:')
        print('T1 =', T1_start)
        T2_start = GP.T_high(Tmelt, sig_start)
        print('T2 =', T2_start)
        self.WriteResults(status = f"Calculations for the first point L = {self.L_array[0]} started")
            
        L1, T1, sol1, liq1 = LAMMPS.data_append(T1_start, self.L_array[0], ksol, bsol, kliq, bliq)
        print('Results: L, T, number of solid outcomes, number of liquid outcomes')
        print(L1, T1, sol1, liq1)
        L2, T2, sol2, liq2 = LAMMPS.data_append(T2_start, self.L_array[0], ksol, bsol, kliq, bliq)
        print(L2, T2, sol2, liq2)

        while True:
            if sol1 == 0 and sol2 == 0:
                Tmelt = Tmelt - 2.5*sig_start
                L2, T2, sol2, liq2 = L1, T1, sol1, liq1
                T1 = GP.T_low(Tmelt, sig_start)
                print('low:',T1,T2)
                L1, T1, sol1, liq1 = LAMMPS.data_append(T1, self.L_array[0], ksol, bsol, kliq, bliq)
            elif liq1 == 0 and liq2 == 0:
                Tmelt = Tmelt + 2.5*sig_start
                L1, T1, sol1, liq1 = L2, T2, sol2, liq2
                T2 = GP.T_high(Tmelt, sig_start)
                print('high:',T1,T2)
                L2, T2, sol2, liq2 = LAMMPS.data_append(T2, self.L_array[0], ksol, bsol, kliq, bliq)
            else:
                print('okay')
                break

        melt.AddDataPoint(L1, T1, sol1, liq1)
        LAMMPS.temp_sl(L1, T1, sol1, liq1, Dict_new)

        melt.AddDataPoint(L2, T2, sol2, liq2)
        LAMMPS.temp_sl(L2, T2, sol2, liq2, Dict_new)

        with open(name, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([L1, T1, sol1, liq1])
            writer.writerow([L2, T2, sol2, liq2])
        return Dict_new

    def FirstRead(self, LAMMPS, melt, name):
        Dict_new = {}
        data = pd.read_csv(name, header=None)
        for i in range(len(data)):
            melt.AddDataPoint(data.iloc[i,0], data.iloc[i,1], data.iloc[i,2], data.iloc[i,3])
            LAMMPS.temp_sl(data.iloc[i,0], data.iloc[i,1], data.iloc[i,2], data.iloc[i,3], Dict_new)
        return Dict_new
    
    def MainFunc(self,name):
        Tmelt = self.Tmelt
        L_array = self.L_array
        N_array = self.N_array
        
        L_train = np.array([L_array[0]])
        N_train = np.array([N_array[0]])
        var_T_start = np.array([1e+6])
        s_0 = random.randint(1000,10000)
        sig_start = s_0/N_train
        var_sig_start = np.array([1e+6])
        N_star = np.array([np.inf])
        N_test = np.linspace(200, 60000, 10000)

        sig_dict = {}
        log_likelihood_dict = {}
        N_train_dict = {}
        T_train_dict = {}

        melt = integrator.Bayesian()
        GP = GaussianProcess()
        LAMMPS = LammpsCalculations()
        
        # Можно закомментить строку ниже, раскомментить следующую
        ksol, bsol, kliq, bliq = self.Lines(LAMMPS, Tmelt)
        #ksol, bsol, kliq, bliq = 0.00016887229916094434, -3.426957680188676, 0.00018372423139186361, -3.3184970819804107
        
        # Тут тоже как выше можно закомментить 
        Dict_new = self.FirstAdd(GP, LAMMPS, melt, name, sig_start, ksol, bsol, kliq, bliq, Tmelt)
        #Dict_new = self.FirstRead(LAMMPS, melt, name)
        print('MD data:')
        print(Dict_new)

        tic = time.perf_counter()
        self.WriteResults("Gaussian process is working")
        v = 0
        condition = 1.0 # Точность расчетов 
        print('Desired accuracy:', condition)
        
        N_train = np.array([])
        T_train = np.array([])
        var_temp = np.array([])
            
        while True:
            

            for L in Dict_new:
                if L not in L_train:
                    L_train = np.append(L_train, L)
            print("L_train:",L_train)

            for i in range(len(L_train)):
                for j in range(len(L_array)):
                    if L_array[j] == L_train[i]:
                        N_train = np.append(N_train, N_array[j])
                        #print('N_train:',N_train)
                        N_train_dict.update({L_array[j]: N_array[j]})
                        #print('N_train_dict:',N_train_dict)
            print('N_train:',N_train)

            res_s0 = optimize.minimize(GP.sig0, x0=3000, args=(N_train, L_train, melt), method='Nelder-Mead')
            s_0 = res_s0.x
            print("sigma_0:", s_0)

            sig_array = s_0/N_train
            #print(sig_array)

            for L in Dict_new:
                sig_dict.update({L: s_0/N_train_dict[L]})
                results = melt.T_melt(L, float(sig_dict[L]), 1e-7)
                T_train = np.append(T_train, results[1])
                T_train_dict.update({L: results[1]})
                #print('T_train_dict:',T_train_dict)
                var_temp = np.append(var_temp, results[2])
            print('T_train_dict:',T_train_dict)
                        
            print('sigma:',sig_dict)
            print('T_train:', T_train)
            print('var_temp:',var_temp)

            x0 = [10., 1000.]
            res = optimize.minimize(GP.log_p_gauss, x0, args=(N_train, T_train, var_temp), method='Nelder-Mead')
            thN = res.x[0]
            sig_f = res.x[1]
            print('hyperparameters thN:',thN, 'sig_f:', sig_f)

            self.T_needed = GP.f_mean(N_train, T_train, N_star, var_temp, thN, sig_f)
            self.var_needed = np.sqrt(GP.var(N_train, N_star, var_temp, thN, sig_f))
            
            print('T_inf =', float(self.T_needed))
            print('delta_T_inf =', float(self.var_needed))
            
            f_star = GP.f_mean(N_train, T_train, N_test, var_temp, thN, sig_f)
            Var_f_star = GP.var(N_train, N_test, var_temp, thN, sig_f)

            sqrtVar_f_star = np.sqrt(Var_f_star)

            f_und = f_star - sqrtVar_f_star
            f_up = f_star + sqrtVar_f_star

            y1 = []
            y2 = []
            for i in range(len(f_und)):
                y1.append(f_und[i][0])
                y2.append(f_up[i][0])
                
            fig, ax = plt.subplots(figsize=(8,5))

            ax.plot(1/N_test, f_star, color = 'blue', label = 'mean value')
            ax.scatter(1/N_train, T_train)
            
            ax.fill_between(1/N_test, y1, y2, alpha = 0.1, color = 'blue',linewidth = 1, linestyle = '-', label='+-sigma')

            ax.set_xlim(0, 0.005)
            ax.set_xticks(1/N_train)
            ax.set_xticklabels(N_train, rotation=335)
            
            plt.xlabel('N, atoms')
            plt.ylabel('T, K')
            plt.title('Al EAM')

            if v == 0:
                ax.yaxis.set_major_locator(ticker.MultipleLocator(50))
                ax.yaxis.set_minor_locator(ticker.MultipleLocator(10))
            else:
                ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
                ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))

            ax.grid(which='major', color = 'k', linewidth = 0.7)

            ax.minorticks_on()

            ax.grid(which='minor', color = 'gray', linestyle = ':')

            plt.errorbar(1/N_train, T_train, yerr=np.sqrt(var_temp), fmt="o", color = 'blue')

            plt.legend()
            plt.savefig('GP.png', facecolor='white', dpi=200)
            plt.show()
            self.WriteResults(status = f"Gaussian process is working. The number of iterations is {v+1}")
            
            if self.var_needed <= condition:
                break
                
            L_next, arg_next, N_next = GP.add_L(L_array, N_array, L_train, N_train, N_star, var_temp, s_0, thN, sig_f)
            N_next = np.array([N_next])
            print('L_next:',L_next)

            if N_next in N_train:
                for i in range(len(N_train)):
                    if N_train[i] == N_next:
                        T_next = T_train[i]
            else:
                T_next = GP.f_mean(N_train, T_train, N_next, var_temp, thN, sig_f)
                if T_next > T_train[0] + 10*sig_start or T_next < T_train[0] - 10*sig_start:
                    T_next = T_train[0]
            print('T_next:', float(T_next))

            T1 = GP.T_low(T_next, s_0/N_next)
            print('Temperatures of MD for the next L:')
            print('T1:', T1)
            T2 = GP.T_high(T_next, s_0/N_next)
            print('T2:', T2)

            L1, T1, sol1, liq1 = LAMMPS.data_append(T1, L_next, ksol, bsol, kliq, bliq)
            print('Results: L, T, number of solid outcomes, number of liquid outcomes')
            print(L1, T1, sol1, liq1)
            L2, T2, sol2, liq2 = LAMMPS.data_append(T2, L_next, ksol, bsol, kliq, bliq)
            print(L2, T2, sol2, liq2)

            if L_next not in Dict_new:
                while True:
                    if sol1 == 0 and sol2 == 0:
                        T_next = T_next - 2.5*s_0/N_next
                        L2, T2, sol2, liq2 = L1, T1, sol1, liq1
                        T1 = GP.T_low(T_next, s_0/N_next)
                        print('low:',T1,T2)
                        L1, T1, sol1, liq1 = LAMMPS.data_append(T1, L_next, ksol, bsol, kliq, bliq)
                    elif liq1 == 0 and liq2 == 0:
                        T_next = T_next + 2.5*s_0/N_next
                        L1, T1, sol1, liq1 = L2, T2, sol2, liq2
                        T2 = GP.T_high(T_next, s_0/N_next)
                        print('high:',T1,T2)
                        L2, T2, sol2, liq2 = LAMMPS.data_append(T2, L_next, ksol, bsol, kliq, bliq)
                    elif (liq1 == 1 and liq2 == 0) or (sol1 == 0 and sol2 == 1):
                        melt.AddDataPoint(L1, T1, sol1, liq1)
                        LAMMPS.temp_sl(L1, T1, sol1, liq1, Dict_new)
                        melt.AddDataPoint(L2, T2, sol2, liq2)
                        LAMMPS.temp_sl(L2, T2, sol2, liq2, Dict_new)
                        L1, T1, sol1, liq1 = LAMMPS.data_append(T1, L_next, ksol, bsol, kliq, bliq)
                        L2, T2, sol2, liq2 = LAMMPS.data_append(T2, L_next, ksol, bsol, kliq, bliq)
                    else:
                        print('okay')
                        break

            melt.AddDataPoint(L1, T1, sol1, liq1)
            LAMMPS.temp_sl(L1, T1, sol1, liq1, Dict_new)

            melt.AddDataPoint(L2, T2, sol2, liq2)
            LAMMPS.temp_sl(L2, T2, sol2, liq2, Dict_new)

            with open(name, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([L1, T1, sol1, liq1])
                writer.writerow([L2, T2, sol2, liq2])
            
            print('MD data:')
            print(Dict_new)
            v += 1


        toc = time.perf_counter()
        print(Dict_new)
        print('T_inf =', float(self.T_needed))
        print('Var_T_inf =', float(self.var_needed))
        print(f"{toc - tic:0.4f} seconds")
        
        self.WriteResults("Gaussian process finished")
        




