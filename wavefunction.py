import psi4
import numpy as np
from kshelper import diag_H,Timer, ACDIIS,printHeader
import time

mol = psi4.geometry("""
  C    1.4250724    0.0644040   -0.1137090
  H    2.5063157    0.1844071   -0.1159750
  C    0.7262282    0.1039660   -1.2602414
  H    1.2272923    0.2717911   -2.2110146
  C   -0.7262282   -0.1039660   -1.2602414
  H   -1.2272923   -0.2717911   -2.2110146
  C   -1.4250724   -0.0644040   -0.1137090
  H   -2.5063157   -0.1844071   -0.1159750
  C   -0.7318022    0.2394181    1.1953283
  H   -0.7633292    1.3299694    1.3618674
  H   -1.2721104   -0.2098681    2.0368776
  C    0.7318022   -0.2394181    1.1953283
  H    0.7633292   -1.3299694    1.3618674
  H    1.2721104    0.2098681    2.0368776
symmetry C1
noreorient
nocom
""")
psi4.set_options({'basis':        '6-311+G*',
                  'scf_type':     'pk',
                  'reference':    'uhf',
                  'mp2_type':     'conv',
                  'e_convergence': 1e-8,
                  'd_convergence': 1e-8})

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
    "LOC_SUB"  : [0, 1, 2, 3, 4, 5]}

orbs   = [5]
occs   = [0.0]
freeze = ["T"]
spin   = ["b"] 
ovl    = ["T"]
# Get the SCF wavefunction & energies
scf_e, scf_wfn = psi4.energy('scf', return_wfn=True)

nalpha = scf_wfn.nalpha()
nbeta  = scf_wfn.nbeta()
nbf    = scf_wfn.nso()

occa = np.zeros(nbf,dtype=np.float)
occb = np.zeros(nbf,dtype=np.float)
occa[:nalpha] = 1.0
occb[:nbeta] = 1.0

OCCA = psi4.core.Vector(nbf)
OCCB = psi4.core.Vector(nbf)
OCCA.np[:] = occa
OCCB.np[:] = occb

Ca = np.asarray(scf_wfn.Ca())
Cb = np.asarray(scf_wfn.Cb())

epsa = np.asarray(scf_wfn.epsilon_a())
epsb = np.asarray(scf_wfn.epsilon_b())

occa = np.asarray(scf_wfn.occupation_a())
occb = np.asarray(scf_wfn.occupation_b())

mw = psi4.core.MoldenWriter(scf_wfn)
mw.write('Neutral.molden',scf_wfn.Ca(),scf_wfn.Cb(),scf_wfn.epsilon_a(),scf_wfn.epsilon_b(),OCCA,OCCB,True)

#LOCALISATION
loc_sub = np.array(options["LOC_SUB"],dtype=np.int)
locCa = psi4.core.Matrix(scf_wfn.nso(),len(loc_sub))
locCb = psi4.core.Matrix(scf_wfn.nso(),len(loc_sub))

locCa.np[:] = np.copy(Ca[:,loc_sub])
locCb.np[:] = np.copy(Cb[:,loc_sub])

LocalA = psi4.core.Localizer.build("PIPEK_MEZEY", scf_wfn.basisset(), locCa)
LocalB = psi4.core.Localizer.build("PIPEK_MEZEY", scf_wfn.basisset(), locCb)

LocalA.localize()
LocalB.localize()

Ca[:,loc_sub] = LocalA.L
Cb[:,loc_sub] = LocalB.L

OCCA = psi4.core.Vector(nbf)
OCCB = psi4.core.Vector(nbf)
OCCA.np[:] = occa
OCCB.np[:] = occb

Ca = np.asarray(scf_wfn.Ca())
Cb = np.asarray(scf_wfn.Cb())

epsa = np.asarray(scf_wfn.epsilon_a())
epsb = np.asarray(scf_wfn.epsilon_b())

occa = np.asarray(scf_wfn.occupation_a())
occb = np.asarray(scf_wfn.occupation_b())

mw = psi4.core.MoldenWriter(scf_wfn)
mw.write('Neutral_loc.molden',scf_wfn.Ca(),scf_wfn.Cb(),scf_wfn.epsilon_a(),scf_wfn.epsilon_b(),OCCA,OCCB,True)

#CORE IONISATION

orbitals = []

lens = [len(x) for x in [orbs,occs,freeze,spin,ovl]]
if  len(list((set(lens))))>1:
    raise Exception("Input arrays have inconsistent length"+" ".join(str(lens)))
for i in range(len(orbs)):
    orbitals.append({"orb" : orbs[i],"spin": spin[i].lower(),"occ" : occs[i], "frz" : freeze[i]=="T","DoOvl":ovl[i] == "T" })

#Ca = np.load("gstate_sorbs.npz")["Ca"]
#Cb = np.load("gstate_sorbs.npz")["Cb"]

for i in orbitals:
    if i["spin"]=="b":
        i["C"] = Cb[:,i["orb"]]
    elif i["spin"]=="a":
            i["C"] = Ca[:,i["orb"]]
    else:
        raise Exception("Orbital has non a/b spin!")

mints = psi4.core.MintsHelper(scf_wfn.basisset())
S = np.asarray(mints.ao_overlap())
T = np.asarray(mints.ao_kinetic())
V = np.asarray(mints.ao_potential())
H = np.zeros((mints.nbf(),mints.nbf()))

H = T+V
A = mints.ao_overlap()
A.power(-0.5,1.e-16)
A = np.asarray(A)

Enuc = mol.nuclear_repulsion_energy()
Eold = 0.0
SCF_E = 100.0

occa = np.zeros(Ca.shape[0])
occb = np.zeros(Cb.shape[0])
occa[:nalpha] = 1
occa[:nbeta]  = 1
Cocca = psi4.core.Matrix(nbf, nbf)
Coccb = psi4.core.Matrix(nbf, nbf)
Cocca.np[:]  = Ca
Coccb.np[:]  = Cb
#initial density
for i in orbitals:
    if i["spin"]=="b":
        ovl = np.abs(np.einsum('m,nj,mn->j',i["C"],Coccb,S))
        if i["orb"] != np.argmax(ovl):
            print ("index changed from {:d} to {:d}".format(i["orb"],np.argmax(ovl)))
            i["orb"] = np.argmax(ovl)

        """
        Set occupation and overlap
        """
        i["ovl"] = np.max(ovl)
        occb[i["orb"]] = i["occ"]

for i in range(nbf):
    Cocca.np[:,i] *= np.sqrt(occa[i])
    Coccb.np[:,i] *= np.sqrt(occb[i])

Da = Cocca.np @ Cocca.np.T
Db = Coccb.np @ Coccb.np.T
jk = psi4.core.JK.build(scf_wfn.basisset(),jk_type=psi4.core.get_local_option("SCF","SCF_TYPE"))
glob_mem = psi4.core.get_memory()/8
jk.set_memory(int(glob_mem*0.6))
jk.initialize()
jk.C_left_add(Cocca)
jk.C_left_add(Coccb)

Da_m = psi4.core.Matrix(nbf,nbf)
Db_m = psi4.core.Matrix(nbf,nbf)

diis = ACDIIS(max_vec=options["DIIS_LEN"])
printHeader("Starting SCF:",2)
psi4.core.print_out("""{:>10} {:8.2E}
{:>10} {:8.2E}
{:>10} {:8.4f}
{:>10} {:8.2E}
{:>10} {:8.4f}
{:>10} {:8d}
{:>10} {:8d}\n
""".format(
    "E_CONV:",options["E_CONV"],
    "D_CONV:",options["D_CONV"],
    "DAMP:",options["GAMMA"],
    "DIIS_EPS:",options["DIIS_EPS"],
    "VSHIFT:",options["VSHIFT"],
    "MAXITER:",options["MAXITER"],
    "DIIS_LEN:",options["DIIS_LEN"],
    "DIIS_MODE:",options["DIIS_MODE"]))

psi4.core.print_out("\nInitial orbital occupation pattern:\n\n")
psi4.core.print_out("Index|Spin|Occ|Ovl|Freeze\n"+25*"-")
for i in orbitals:
        psi4.core.print_out("\n{:^5}|{:^4}|{:^3}|{:^3}|{:^6}".format(i["orb"],i["spin"],i["occ"],'Yes' if i["DoOvl"] else 'No','Yes' if i["frz"] else 'No'))
psi4.core.print_out("\n\n")

psi4.core.print_out(("{:^3} {:^14} {:^11} {:^11} {:^11} {:^5} {:^5} | {:^"+str(len(orbitals)*5)+"}| {:^11} {:^5}\n").format("#IT", "Escf",
         "dEscf","Derror","DIIS-E","na","nb",
         "OVL","MIX","Time"))
psi4.core.print_out("="*(87+5*len(orbitals))+"\n")

diis_counter = 0
myTimer = Timer()
for SCF_ITER in range(1, options["MAXITER"]+ 1):
    myTimer.addStart("SCF")

    myTimer.addStart("JK")
    jk.compute()
    myTimer.addEnd("JK")

    myTimer.addStart("buildFock")
    Da_m.np[:] = Da
    Db_m.np[:] = Db

    Ja = np.asarray(jk.J()[0])
    Jb = np.asarray(jk.J()[1])
    Ka = np.asarray(jk.K()[0])
    Kb = np.asarray(jk.K()[1])
    
    if SCF_ITER>1 :
        FaOld = np.copy(Fa)
        FbOld = np.copy(Fb)

    Fa = H +  (Ja + Jb) - Ka
    Fb = H +  (Ja + Jb) - Kb

    myTimer.addEnd("buildFock")

    myTimer.addStart("Freeze")
    FMOa = Ca.T @ Fa @ Ca
    FMOb = Cb.T @ Fb @ Cb

    CaInv = np.linalg.inv(Ca)
    CbInv = np.linalg.inv(Cb)

    for i in orbitals:
        if i['frz'] == True:
            if i["spin"]=="b":
                idx = i["orb"]
                FMOb[idx,:idx]     = 0.0
                FMOb[idx,(idx+1):] = 0.0
                FMOb[:idx,idx]     = 0.0
                FMOb[(idx+1):,idx] = 0.0
            elif i["spin"]=="a":
                FMOa[idx,:idx]     = 0.0
                FMOa[idx,(idx+1):] = 0.0
                FMOa[:idx,idx]     = 0.0
                FMOa[(idx+1):,idx] = 0.0
    """
    VSHIFT
    """
    idxs = [c for c,x in enumerate(occa) if (x ==0.0) and (c>=nalpha)]
    FMOa[idxs,idxs] += options["VSHIFT"]
    idxs = [c for c,x in enumerate(occb) if (x ==0.0) and (c>=nbeta)]
    FMOb[idxs,idxs] += options["VSHIFT"]

    Fa = CaInv.T @ FMOa @ CaInv
    Fb = CbInv.T @ FMOb @ CbInv

    myTimer.addEnd("Freeze")

    diisa_e = np.ravel(A.T@(Fa@Da@S - S@Da@Fa)@A)
    diisb_e = np.ravel(A.T@(Fb@Db@S - S@Db@Fb)@A)
    diis.add(Fa,Fb,Da,Db,np.concatenate((diisa_e,diisb_e)))

    if ("DIIS" in options["MIXMODE"]) and (SCF_ITER>1):
        (Fa,Fb) = diis.extrapolate(DIISError)
        diis_counter += 1
        if (diis_counter >= 2*options["DIIS_LEN"]):
            diis.reset()
            diis_counter = 0
            psi4.core.print_out("Resetting DIIS\n")

    elif (options["MIXMODE"] == "DAMP") and (SCF_ITER>1):    
        Fa = (1-options["GAMMA"]) * np.copy(Fa) + (options["GAMMA"]) * FaOld
        Fb = (1-options["GAMMA"]) * np.copy(Fb) + (options["GAMMA"]) * FbOld
    
    myTimer.addEnd("MIX")

    myTimer.addStart("calcE")

    one_electron_E   = np.sum(Da * H)
    one_electron_E  += np.sum(Db * H)
    two_electron_E   = np.sum(Da * (Ja+Jb-Ka))
    two_electron_E  += np.sum(Db * (Ja+Jb-Kb))

    SCF_E  = 0.0
    SCF_E += Enuc
    SCF_E += one_electron_E
    SCF_E += 0.5 * two_electron_E
    myTimer.addEnd("calcE")

    myTimer.addStart("Diag")
    Ca,epsa = diag_H(Fa, A)
    Cb,epsb = diag_H(Fb, A)
    myTimer.addEnd("Diag")

    DaOld = np.copy(Da)
    DbOld = np.copy(Db)

    """
    New ornitals obatained set occupation numbers
    """
    myTimer.addStart("SetOcc")

    Cocca.np[:]  = Ca
    Coccb.np[:]  = Cb

    occa[:] = 0.0
    occb[:] = 0.0

    occa[:nalpha] = 1.0 #standard aufbau principle occupation
    occb[:nbeta]  = 1.0

    for i in orbitals:
        if i["spin"]=="b":
            """
            Overalp
            """
            #calculate the Overlapp with all other orbitals
            ovl = np.abs(np.einsum('m,nj,mn->j',i["C"],Coccb,S))
            #User wants to switch the index if higher overlap is found
            if i["DoOvl"] ==True :
                if i["orb"] != np.argmax(ovl):
                    i["orb"] = np.argmax(ovl)
                i["ovl"] = np.max(ovl)

            else:
                i["ovl"] = ovl[i["orb"]]
            occb[i["orb"]] = i["occ"]

        elif i["spin"]=="a":
            """
            Check if this is still the largest overlap
            """

            ovl = np.abs(np.einsum('m,nj,mn->j',i["C"],Cocca,S))
            if i["DoOvl"] ==True :
                if i["orb"] != np.argmax(ovl):
                    i["orb"] = np.argmax(ovl) # set index to the highest overlap
                i["ovl"] = np.max(ovl)
            else:
                i["ovl"] = ovl[i["orb"]]
            #Modify the occupation vector
            occa[i["orb"]] = i["occ"]

    for i in range(nbf):
        Cocca.np[:,i] *= np.sqrt(occa[i])
        Coccb.np[:,i] *= np.sqrt(occb[i])

    Da     = Cocca.np @ Cocca.np.T
    Db     = Coccb.np @ Coccb.np.T
    myTimer.addEnd("SetOcc")

    DError = (np.sum((DaOld-Da)**2)**0.5 + np.sum((DbOld-Db)**2)**0.5)
    EError = (SCF_E - Eold)
    DIISError = (np.sum(diisa_e**2)**0.5 + np.sum(diisb_e**2)**0.5)

    myTimer.addEnd("SCF")
    psi4.core.print_out(("{:3d} {:14.8f} {:11.3E} {:11.3E} {:11.3E} {:5.1f} {:5.1f} | "+"{:4.2f} "*len(orbitals)+"| {:^11} {:5.2f} {:2d}  \n").format(
        SCF_ITER,
        SCF_E,
        EError,
        DError,
        DIISError,
        np.sum(Da*S),
        np.sum(Db*S),
        *[x["ovl"] for x in orbitals],
        options["MIXMODE"],
        myTimer.getTime("SCF"),
        len(diis.Fa)))

    if (abs(DIISError) < options["DIIS_EPS"]):
        options["MIXMODE"] = options["DIIS_MODE"]
    else:
        options["MIXMODE"] = "DAMP"

    if (abs(EError) < options["E_CONV"]) and (abs(DError)<options["D_CONV"]):
        if (options["VSHIFT"] != 0.0):
            psi4.core.print_out("Converged but Vshift was on... removing Vshift..\n")
            options["VSHIFT"] = 0.0
        else:
            break

    Eold = SCF_E

    if SCF_ITER == options["MAXITER"]:
         psi4.core.clean()
         np.savez('exstate_orbs',Ca=Ca,Cb=Cb,occa=occa,occb=occb,epsa=epsa,epsb=epsb,orbitals=orbitals)
         raise Exception("Maximum number of SCF cycles exceeded.")

psi4.core.print_out("\n\n{:>20} {:12.8f} [Ha] \n".format("FINAL EX SCF ENERGY:",SCF_E))
gsE = scf_e
if gsE!=0.0:
    psi4.core.print_out("{:>20} {:12.8f} [Ha] \n".format("EXCITATION ENERGY:",SCF_E-gsE))
    psi4.core.print_out("{:>20} {:12.8f} [eV] \n\n".format("EXCITATION ENERGY:",(SCF_E-gsE)*27.211385))    
psi4.core.print_out("\nFinal orbital occupation pattern:\n\n")
psi4.core.print_out("Index|Spin|Occ|Ovl|Freeze|Comment\n"+34*"-")
for i in orbitals:
    Comment = "-"
    if i["DoOvl"]:
        psi4.core.print_out("\n{:^5}|{:^4}|{:^3}|{:^3}|{:^6}|{:^7}".format(i["orb"],i["spin"],i["occ"],'Yes' if i["DoOvl"] else 'No','Yes' if i["frz"] else 'No',Comment))
    else:
        if i["spin"]=="b":
            #calculate the Overlap with all other orbitals
            ovl = np.abs(np.einsum('m,nj,mn->j',i["C"],Coccb,S))
            idx = np.argmax(ovl)
        elif  i["spin"]=="a":
            ovl = np.abs(np.einsum('m,nj,mn->j',i["C"],Cocca,S))
            idx = np.argmax(ovl)
        Comment = " Found by overlap"
        psi4.core.print_out("\n{:^5}|{:^4}|{:^3}|{:^3}|{:^6}|{:^7}".format(idx,i["spin"],i["occ"],'Yes' if i["DoOvl"] else 'No','Yes' if i["frz"] else 'No',Comment))


OCCA = psi4.core.Vector(nbf)
OCCB = psi4.core.Vector(nbf)
OCCA.np[:] = occa
OCCB.np[:] = occb
scf_wfn.Ca().np[:] = Ca
scf_wfn.Cb().np[:] = Cb
scf_wfn.epsilon_a().np[:] = epsa
scf_wfn.epsilon_b().np[:] = epsb
scf_wfn.occupation_a().np[:] = occa
scf_wfn.occupation_b().np[:] = occb
mw = psi4.core.MoldenWriter(scf_wfn)
mw.write('core_hole.molden',scf_wfn.Ca(),scf_wfn.Cb(),scf_wfn.epsilon_a(),scf_wfn.epsilon_b(),OCCA,OCCB,True)
