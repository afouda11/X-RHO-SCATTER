"""

This code is a modified version of the psixas ksex.py code that also gives the option to do HF as well as DFT
https://github.com/Masterluke87/psixas

"""

import psi4
import numpy as np
from kshelper import diag_H,Timer, ACDIIS,printHeader
import time


def ground_state(basis, dft=False, func=None):
    
    psi4.set_options({'scf_type':      'df',
                      'reference':     'uhf',
                      'e_convergence': 1e-8,
                      'd_convergence': 1e-8,
                  })

    # Get the SCF wavefunction & energies
    global gsE
    if dft == False:
        gsE, wfn = psi4.energy('scf', return_wfn=True)
    if dft == True:
        gsE, wfn = psi4.energy(func, return_wfn=True)

    nalpha = wfn.nalpha()
    nbeta  = wfn.nbeta()
    nbf    = wfn.nso()

    occa = np.zeros(nbf,dtype=np.float)
    occb = np.zeros(nbf,dtype=np.float)
    occa[:nalpha] = 1.0
    occb[:nbeta] = 1.0

    OCCA = psi4.core.Vector(nbf)
    OCCB = psi4.core.Vector(nbf)
    OCCA.np[:] = occa
    OCCB.np[:] = occb

    Ca = np.asarray(wfn.Ca())
    Cb = np.asarray(wfn.Cb())

    epsa = np.asarray(wfn.epsilon_a())
    epsb = np.asarray(wfn.epsilon_b())

    occa = np.asarray(wfn.occupation_a())
    occb = np.asarray(wfn.occupation_b())

    mw = psi4.core.MoldenWriter(wfn)

    if dft == False:
        mw.write('Neutral_HF.molden',wfn.Ca(),wfn.Cb(),wfn.epsilon_a(),wfn.epsilon_b(),OCCA,OCCB,True)
        np.savez('hf_gsorbs',Ca=Ca,Cb=Cb,occa=occa,occb=occb,epsa=epsa,epsb=epsb)

    if dft == True:
        mw.write('Neutral_DFT.molden',wfn.Ca(),wfn.Cb(),wfn.epsilon_a(),wfn.epsilon_b(),OCCA,OCCB,True)
        np.savez('dft_gsorbs',Ca=Ca,Cb=Cb,occa=occa,occb=occb,epsa=epsa,epsb=epsb)

    return wfn 

def localize(wfn, loc_sub, dft):
    #loc_sub = np.array(options["LOC_SUB"],dtype=np.int)
    #wfn     = psi4.core.Wavefunction.build(mol,psi4.core.get_global_option('BASIS'))
    nbf    = wfn.nso()

    if dft == True:
        sup = psi4.driver.dft.build_superfunctional(func, False)[0]
        sup.set_deriv(2)
        sup.allocate()
        uhf   = psi4.core.UHF(wfn,sup)
        Ca = np.load("dft_gsorbs.npz")["Ca"]
        Cb = np.load("dft_gsorbs.npz")["Cb"]
        occa = np.load("dft_gsorbs.npz")["occa"]
        occb = np.load("dft_gsorbs.npz")["occb"]
        epsa = np.load("dft_gsorbs.npz")["epsa"]
        epsb = np.load("dft_gsorbs.npz")["epsb"]

    if dft == False:
        Ca = np.load("hf_gsorbs.npz")["Ca"]
        Cb = np.load("hf_gsorbs.npz")["Cb"]
        occa = np.load("hf_gsorbs.npz")["occa"]
        occb = np.load("hf_gsorbs.npz")["occb"]
        epsa = np.load("hf_gsorbs.npz")["epsa"]
        epsb = np.load("hf_gsorbs.npz")["epsb"]

    locCa = psi4.core.Matrix(wfn.nso(),len(loc_sub))
    locCb = psi4.core.Matrix(wfn.nso(),len(loc_sub))

    locCa.np[:] = np.copy(Ca[:,loc_sub])
    locCb.np[:] = np.copy(Cb[:,loc_sub])

    LocalA = psi4.core.Localizer.build("PIPEK_MEZEY", wfn.basisset(), locCa)
    LocalB = psi4.core.Localizer.build("PIPEK_MEZEY", wfn.basisset(), locCb)

    LocalA.localize()
    LocalB.localize()

    Ca[:,loc_sub] = LocalA.L
    Cb[:,loc_sub] = LocalB.L

    if dft == True:
        np.savez('dft_loc_gsorbs',Ca=Ca,Cb=Cb,occa=occa,occb=occb)
        psi4.core.print_out("Localized Orbitals written\n")

    if dft == False:
        np.savez('hf_loc_gsorbs',Ca=Ca,Cb=Cb,occa=occa,occb=occb)
        psi4.core.print_out("Localized Orbitals written\n")

    OCCA = psi4.core.Vector(nbf)
    OCCB = psi4.core.Vector(nbf)
    OCCA.np[:] = occa
    OCCB.np[:] = occb

    if dft == False:

        wfn.Ca().np[:] = Ca
        wfn.Cb().np[:] = Cb

        wfn.epsilon_a().np[:] = epsa
        wfn.epsilon_b().np[:] = epsb

        wfn.occupation_a().np[:] = occa
        wfn.occupation_b().np[:] = occb

        mw = psi4.core.MoldenWriter(wfn)
        mw.write('Neutral_HF_loc.molden',wfn.Ca(),wfn.Cb(),wfn.epsilon_a(),wfn.epsilon_b(),OCCA,OCCB,True)

    if dft == True:
        uhf.Ca().np[:] = Ca
        uhf.Cb().np[:] = Cb

        uhf.epsilon_a().np[:] = epsa
        uhf.epsilon_b().np[:] = epsb

        uhf.occupation_a().np[:] = occa
        uhf.occupation_b().np[:] = occb

        mw = psi4.core.MoldenWriter(uhf)
        mw.write('Neutral_DFT_loc.molden',uhf.Ca(),uhf.Cb(),uhf.epsilon_a(),uhf.epsilon_b(),OCCA,OCCB,True)

def non_aufbau_state(orbs, occs, freeze, spin, ovl, dft, func, mol, scf_wfn, **options):
    orbitals = []

    lens = [len(x) for x in [orbs,occs,freeze,spin,ovl]]
    if  len(list((set(lens))))>1:
        raise Exception("Input arrays have inconsistent length"+" ".join(str(lens)))
    for i in range(len(orbs)):
        orbitals.append({"orb" : orbs[i],"spin": spin[i].lower(),"occ" : occs[i], "frz" : freeze[i]=="T","DoOvl":ovl[i] == "T" })

    if dft == True:
        Ca = np.load("dft_loc_gsorbs.npz")["Ca"]
        Cb = np.load("dft_loc_gsorbs.npz")["Cb"]

    if dft == False:
        Ca = np.load("hf_loc_gsorbs.npz")["Ca"]
        Cb = np.load("hf_loc_gsorbs.npz")["Cb"]

    for i in orbitals:
        if i["spin"]=="b":
            i["C"] = Cb[:,i["orb"]]
        elif i["spin"]=="a":
                i["C"] = Ca[:,i["orb"]]
        else:
            raise Exception("Orbital has non a/b spin!")


    wfn    = psi4.core.Wavefunction.build(mol, psi4.core.get_global_option('basis'))
    mints = psi4.core.MintsHelper(wfn.basisset())
    S = np.asarray(mints.ao_overlap())
    T = np.asarray(mints.ao_kinetic())
    V = np.asarray(mints.ao_potential())
    H = np.zeros((mints.nbf(),mints.nbf()))

    H = T+V
    if wfn.basisset().has_ECP():
        ECP = mints.ao_ecp()
        H += ECP

    A = mints.ao_overlap()
    A.power(-0.5,1.e-16)
    A = np.asarray(A)

    Enuc = mol.nuclear_repulsion_energy()
    Eold  =  0.0
    SCF_E = 0.0
    nbf    = wfn.nso()
    nalpha = wfn.nalpha()
    nbeta  = wfn.nbeta()

    if dft == True:
        Va = psi4.core.Matrix(nbf,nbf)
        Vb = psi4.core.Matrix(nbf,nbf)
        sup = psi4.driver.dft.build_superfunctional(func, False)[0]
        sup.allocate()

        Vpot = psi4.core.VBase.build(wfn.basisset(), sup, "UV")
        Vpot.initialize()

        uhf   = psi4.core.UHF(wfn,sup)
        Da_m = psi4.core.Matrix(nbf,nbf)
        Db_m = psi4.core.Matrix(nbf,nbf)

    occa = np.zeros(Ca.shape[0])
    occb = np.zeros(Cb.shape[0])
    occa[:nalpha] = 1
    occb[:nbeta]  = 1
    Cocca = psi4.core.Matrix(nbf, nbf)
    Coccb = psi4.core.Matrix(nbf, nbf)
    Cocca.np[:]  = Ca
    Coccb.np[:]  = Cb

#initial density
    for i in orbitals:
        if i["spin"]=="b":
            overlap = np.abs(np.einsum('m,nj,mn->j',i["C"],Coccb,S))
            if i["orb"] != np.argmax(overlap):
                print ("index changed from {:d} to {:d}".format(i["orb"],np.argmax(overlap)))
                i["orb"] = np.argmax(overlap)

            """
            Set occupation and overlap
            """
            i["ovl"] = np.max(overlap)
            occb[i["orb"]] = i["occ"]
    for i in range(nbf):
        Cocca.np[:,i] *= np.sqrt(occa[i])
        Coccb.np[:,i] *= np.sqrt(occb[i])

    Da     = Cocca.np @ Cocca.np.T
    Db     = Coccb.np @ Coccb.np.T

    jk = psi4.core.JK.build(wfn.basisset())
    glob_mem = psi4.core.get_memory()/8
    jk.set_memory(int(glob_mem*0.6))
    if dft == "T":
        if (sup.is_x_hybrid()):
            jk.set_do_K(True)
        if (sup.is_x_lrc()):
             jk.set_omega(sup.x_omega())
             jk.set_do_wK(True)
    jk.initialize()
    jk.C_left_add(Cocca)
    jk.C_left_add(Coccb)

    jk.compute()
    Ja = np.asarray(jk.J()[0])
    Jb = np.asarray(jk.J()[1])
    Ka = np.asarray(jk.K()[0])
    Kb = np.asarray(jk.K()[1])
    one_electron_E = np.einsum('pq,pq->', (Da + Db), H, optimize=True)

    if dft == False:
        two_electron_E  = 0.5 * np.sum(Da * (Ja+Jb -Ka))
        two_electron_E += 0.5 * np.sum(Db * (Ja+Jb -Kb))
    if dft == True:
        Da_m.np[:] = Da
        Db_m.np[:] = Db
        Vpot.set_D([Da_m,Db_m])
        Vpot.compute_V([Va,Vb])

        coulomb_E       = np.sum(Da * (Ja+Jb))
        coulomb_E      += np.sum(Db * (Ja+Jb))
        exchange_E  = 0.0
        if sup.is_x_hybrid():
            exchange_E -=  sup.x_alpha() * np.sum(Da * Ka)
            exchange_E -=  sup.x_alpha() * np.sum(Da * Kb)
        if sup.is_x_lrc():
            exchange_E -= sup.x_beta() * np.sum(Da * np.asarray(jk.wK()[0]))
            exchange_E -= sup.x_beta() * np.sum(Db * np.asarray(jk.wK()[1]))
        XC_E = Vpot.quadrature_values()["FUNCTIONAL"]

        two_electron_E = 0.5 * (coulomb_E + exchange_E) + XC_E

    SCF_E += Enuc + one_electron_E + two_electron_E

    psi4.core.print_out("{:>20} {:12.8f} [Ha] \n".format("One-Electron Energy:",one_electron_E))
    psi4.core.print_out("{:>20} {:12.8f} [Ha] \n".format("Two-Electron Energy:",two_electron_E))
    psi4.core.print_out("{:>20} {:12.8f} [Ha] \n".format("Total Guess Energy:",SCF_E))

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

        Ja = np.asarray(jk.J()[0])
        Jb = np.asarray(jk.J()[1])

        Ka = np.asarray(jk.K()[0])
        Kb = np.asarray(jk.K()[1])
        
        if dft == True:
            Da_m.np[:] = Da
            Db_m.np[:] = Db
            Vpot.set_D([Da_m,Db_m])
            Vpot.compute_V([Va,Vb])

        if SCF_ITER>1 :
            FaOld = np.copy(Fa)
            FbOld = np.copy(Fb)

        if dft == False:
            Fa = H +  (Ja + Jb) - Ka
            Fb = H +  (Ja + Jb) - Kb

        if dft == True:
            Fa = H + (Ja + Jb) + Va
            Fb = H + (Ja + Jb) + Vb
            if sup.is_x_hybrid():
                 Fa -= sup.x_alpha()* Ka
                 Fb -= sup.x_alpha()* Kb
            if sup.is_x_lrc():
                Fa -= sup.x_beta()*np.asarray(jk.wK()[0])
                Fb -= sup.x_beta()*np.asarray(jk.wK()[1])

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

        one_electron_E = np.einsum('pq,pq->', (Da + Db), H, optimize=True)

        if dft == False:
            two_electron_E  = 0.5 * np.sum(Da * (Ja+Jb -Ka))
            two_electron_E += 0.5 * np.sum(Db * (Ja+Jb -Kb))
        if dft == True:
            Da_m.np[:] = Da
            Db_m.np[:] = Db
            Vpot.set_D([Da_m,Db_m])
            Vpot.compute_V([Va,Vb])

            coulomb_E       = np.sum(Da * (Ja+Jb))
            coulomb_E      += np.sum(Db * (Ja+Jb))
            exchange_E  = 0.0
            if sup.is_x_hybrid():
                exchange_E -=  sup.x_alpha() * np.sum(Da * Ka)
                exchange_E -=  sup.x_alpha() * np.sum(Da * Kb)
            if sup.is_x_lrc():
                exchange_E -= sup.x_beta() * np.sum(Da * np.asarray(jk.wK()[0]))
                exchange_E -= sup.x_beta() * np.sum(Db * np.asarray(jk.wK()[1]))
            XC_E = Vpot.quadrature_values()["FUNCTIONAL"]

            two_electron_E = 0.5 * (coulomb_E + exchange_E) + XC_E
        SCF_E = 0.0
        SCF_E += Enuc + one_electron_E + two_electron_E

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
                overlap = np.abs(np.einsum('m,nj,mn->j',i["C"],Coccb,S))
                #User wants to switch the index if higher overlap is found
                if i["DoOvl"] ==True :
                    if i["orb"] != np.argmax(overlap):
                        i["orb"] = np.argmax(overlap)
                    i["ovl"] = np.max(overlap)

                else:
                    i["ovl"] = overlap[i["orb"]]
                occb[i["orb"]] = i["occ"]

            elif i["spin"]=="a":
                """
                Check if this is still the largest overlap
                """

                overlap = np.abs(np.einsum('m,nj,mn->j',i["C"],Cocca,S))
                if i["DoOvl"] ==True :
                    if i["orb"] != np.argmax(overlap):
                        i["orb"] = np.argmax(overlap) # set index to the highest overlap
                    i["ovl"] = np.max(overlap)
                else:
                    i["ovl"] = overlap[i["orb"]]
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
                overlap = np.abs(np.einsum('m,nj,mn->j',i["C"],Coccb,S))
                idx = np.argmax(overlap)
            elif  i["spin"]=="a":
                overlap = np.abs(np.einsum('m,nj,mn->j',i["C"],Cocca,S))
                idx = np.argmax(overlap)
            Comment = " Found by overlap"
            psi4.core.print_out("\n{:^5}|{:^4}|{:^3}|{:^3}|{:^6}|{:^7}".format(idx,i["spin"],i["occ"],'Yes' if i["DoOvl"] else 'No','Yes' if i["frz"] else 'No',Comment))


    OCCA = psi4.core.Vector(nbf)
    OCCB = psi4.core.Vector(nbf)
    OCCA.np[:] = occa
    OCCB.np[:] = occb
    if dft == False:
        scf_wfn.Ca().np[:] = Ca
        scf_wfn.Cb().np[:] = Cb
        scf_wfn.epsilon_a().np[:] = epsa
        scf_wfn.epsilon_b().np[:] = epsb
        scf_wfn.occupation_a().np[:] = occa
        scf_wfn.occupation_b().np[:] = occb
        mw = psi4.core.MoldenWriter(scf_wfn)
        mw.write('Core_Hole_HF.molden',scf_wfn.Ca(),scf_wfn.Cb(),scf_wfn.epsilon_a(),scf_wfn.epsilon_b(),OCCA,OCCB,True)
    if dft == True:
        uhf.Ca().np[:] = Ca
        uhf.Cb().np[:] = Cb
        uhf.epsilon_a().np[:] = epsa
        uhf.epsilon_b().np[:] = epsb
        uhf.occupation_a().np[:] = occa
        uhf.occupation_b().np[:] = occb
        mw = psi4.core.MoldenWriter(uhf)
        mw.write('Core_Hole_DFT.molden',uhf.Ca(),uhf.Cb(),uhf.epsilon_a(),uhf.epsilon_b(),OCCA,OCCB,True)


