import psi4
import numpy as np
import wf

#specify geometry,
#options below coordinates are keep the molcule fixed in space
#please provide a geometry with its center of mass at the origin
#for the scattering calculation
mol = psi4.geometry("""
        C         1.4225362665    0.0668177861   -0.1118069873
        H         2.5020006041    0.1902163842   -0.1153665161
        C         0.7248577450    0.1081828626   -1.2567194972
        H         1.2268155727    0.2809839700   -2.2045192821
        C        -0.7248577450   -0.1081828626   -1.2567194972
        H        -1.2268155727   -0.2809839700   -2.2045192821
        C        -1.4225362665   -0.0668177861   -0.1118069873
        H        -2.5020006040   -0.1902163843   -0.1153665161
        C        -0.7289141146    0.2427583297    1.1936049221
        H        -0.7560498048    1.3329770153    1.3508934782
        H        -1.2710307622   -0.1951779008    2.0370471334
        C         0.7289141146   -0.2427583297    1.1936049221
        H         0.7560498048   -1.3329770151    1.3508934782
        H         1.2710307621    0.1951779007    2.0370471332
symmetry C1
noreorient
nocom
""")

#PSIXAS options see 
options = {
    "E_CONV"   : 1.0E-6,
    "D_CONV"   : 1.0E-4,
    "GAMMA"    : 0.8,
    "DIIS_EPS" : 0.1,
    "VSHIFT"   : 0.0,
    "MAXITER"  : 100,
    "DIIS_LEN" : 6,
    "DIIS_MODE": "ADIIS+CDIIS",
    "MIXMODE"  : "DAMP",
    "LOC_SUB"  : [0, 1, 2, 3, 4, 5],
    "ORBS"     : [3],  #Options from here need to be the same length
    "OCCS"     : [0.0],#1 to 1 mapping between the options
    "FREEZE"   : ["T"],
    "SPIN"     : ["b"],
    "OVL"      : ["T"]
    }

#psi4 scf options 
psi4.set_options({'basis'         : '6-311+G*',
                  'reference'     : 'uhf',
                  'e_convergence' : 1e-8,
                  'd_convergence' : 1e-8})
psi4.set_num_threads(4)

#do dft? If so, which functional? If not, will do HF
dft    = False
func   = 'b3lyp'

#g-state wf, uses psi4.energy()
scf_wfn = wf.ground_state(dft, func)

#get arrray or orbs to localize, generate loc_wf which overwrites scf_wf
loc_sub = np.array(options["LOC_SUB"],dtype=np.int)
wf.localize(scf_wfn, loc_sub, dft)

#create the core-hole or whatever occupation it is state
wf.non_aufbau_state(dft, func, mol, scf_wfn, **options)

