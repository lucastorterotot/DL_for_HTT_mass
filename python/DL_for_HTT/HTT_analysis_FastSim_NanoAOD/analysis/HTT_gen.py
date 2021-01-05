import DL_for_HTT.HTT_analysis_FastSim_NanoAOD.modules.store_vars as store_vars

def find_HTT(evt):
    ''' Find the Higgs boson in the generated particles,
    check that it decays in tau leptons,
    return the Higgs and the two taus.'''
    Higgs_IDs = [25, 35, 36]
    tau_ID = 15
    gen_taus_from_Higgs_indexes = [k for k in range(int(evt.GetLeaf("nGenPart").GetValue(0))) if abs(evt.GetLeaf("GenPart_pdgId").GetValue(k)) == tau_ID and abs(evt.GetLeaf("GenPart_pdgId").GetValue(int(evt.GetLeaf("GenPart_genPartIdxMother").GetValue(k)))) in Higgs_IDs]
    Higgs_decaying_into_taus_indexes = [int(evt.GetLeaf("GenPart_genPartIdxMother").GetValue(k)) for k in gen_taus_from_Higgs_indexes]
    if len(gen_taus_from_Higgs_indexes) == 2 and Higgs_decaying_into_taus_indexes[0] == Higgs_decaying_into_taus_indexes[1]:
        return Higgs_decaying_into_taus_indexes[0], gen_taus_from_Higgs_indexes[0], gen_taus_from_Higgs_indexes[1]
    # else:
    #     print(gen_taus_from_Higgs_indexes)
    #     print(Higgs_decaying_into_taus_indexes)
    #     import pdb; pdb.set_trace()
    return None, None, None

def determine_gen_channel(evt, tau1, tau2):
    tau1_decays = [(tau1, 15)]
    while 15 in [abs(info[1]) for info in tau1_decays]:
        new_tau1 = [info for info in tau1_decays if abs(info[1]) == 15][0]
        tau1_decays = [(index, evt.GetLeaf("GenPart_pdgId").GetValue(index)) for index in range(int(evt.GetLeaf("nGenPart").GetValue(0))) if
                       int(evt.GetLeaf("GenPart_genPartIdxMother").GetValue(index)) == new_tau1[0]]
    tau2_decays = [(tau2, 15)]
    while 15 in [abs(info[1]) for info in tau2_decays]:
        new_tau2 = [info for info in tau2_decays if abs(info[1]) == 15][0]
        tau2_decays = [(index, evt.GetLeaf("GenPart_pdgId").GetValue(index)) for index in range(int(evt.GetLeaf("nGenPart").GetValue(0))) if
                       int(evt.GetLeaf("GenPart_genPartIdxMother").GetValue(index)) == new_tau2[0]]
    # check with neutrinos in decays
    if 12 in [abs(info[1]) for info in tau1_decays]:
        leg1 = "e"
    elif 14 in [abs(info[1]) for info in tau1_decays]:
        leg1 = "m"
    elif 16 in [abs(info[1]) for info in tau1_decays]:
        leg1 = "t"
    else:
        raise RuntimeError("Tau decay not found!")
    if 12 in [abs(info[1]) for info in tau2_decays]:
        leg2 = "e"
    elif 14 in [abs(info[1]) for info in tau2_decays]:
        leg2 = "m"
    elif 16 in [abs(info[1]) for info in tau2_decays]:
        leg2 = "t"
    else:
        raise RuntimeError("Tau decay not found!")
    channel = leg1+leg2
    gen_vis_tau1, gen_vis_tau2 = new_tau1[0], new_tau2[0]
    # sort legs and channel
    if leg1 == leg2:
        if evt.GetLeaf("GenPart_pt").GetValue(tau2) > evt.GetLeaf("GenPart_pt").GetValue(tau1):
            tau1, tau2 = tau2, tau1
            gen_vis_tau1, gen_vis_tau2 = gen_vis_tau2, gen_vis_tau1
    elif channel in ["tm", "te", "me"]:
        leg1, leg2 = leg2, leg1
        tau1, tau2 = tau2, tau1
        gen_vis_tau1, gen_vis_tau2 = gen_vis_tau2, gen_vis_tau1
    return leg1+leg2, tau1, tau2, gen_vis_tau1, gen_vis_tau2

def get_gen_leg_pt_eta_phi(evt, leg, subchannel = "t"):
    if subchannel == "t":
        for k in range(int(evt.GetLeaf("nGenVisTau").GetValue(0))):
            if evt.GetLeaf("GenVisTau_genPartIdxMother").GetValue(k) == leg or evt.GetLeaf("GenVisTau_genPartIdxMother").GetValue(k) == evt.GetLeaf("GenPart_genPartIdxMother").GetValue(leg):
                return evt.GetLeaf("GenVisTau_pt").GetValue(k), evt.GetLeaf("GenVisTau_eta").GetValue(k), evt.GetLeaf("GenVisTau_phi").GetValue(k)
            print(evt.GetLeaf("GenVisTau_genPartIdxMother").GetValue(k), leg, evt.GetLeaf("GenPart_genPartIdxMother").GetValue(leg))
    if subchannel == "m":
        for k in range(int(evt.GetLeaf("nMuon").GetValue(0))):
            gen_muon_idx = int(evt.GetLeaf("Muon_genPartIdx").GetValue(k))
            if evt.GetLeaf("GenPart_genPartIdxMother").GetValue(gen_muon_idx) == leg or evt.GetLeaf("GenPart_genPartIdxMother").GetValue(gen_muon_idx) == evt.GetLeaf("GenPart_genPartIdxMother").GetValue(leg):
                return evt.GetLeaf("GenPart_pt").GetValue(gen_muon_idx), evt.GetLeaf("GenPart_eta").GetValue(gen_muon_idx), evt.GetLeaf("GenPart_phi").GetValue(gen_muon_idx)
            print( evt.GetLeaf("GenPart_genPartIdxMother").GetValue(gen_muon_idx), leg, evt.GetLeaf("GenPart_genPartIdxMother").GetValue(leg))
    if subchannel == "e":
        for k in range(int(evt.GetLeaf("nElectron").GetValue(0))):
            gen_electron_idx = int(evt.GetLeaf("Electron_genPartIdx").GetValue(k))
            if evt.GetLeaf("GenPart_genPartIdxMother").GetValue(gen_electron_idx) == leg or evt.GetLeaf("GenPart_genPartIdxMother").GetValue(gen_electron_idx) == evt.GetLeaf("GenPart_genPartIdxMother").GetValue(leg):
                return evt.GetLeaf("GenPart_pt").GetValue(gen_electron_idx), evt.GetLeaf("GenPart_eta").GetValue(gen_electron_idx), evt.GetLeaf("GenPart_phi").GetValue(gen_electron_idx)

    return -10, -10, -10
    
                        
def HTT_analysis(evt, verbose = 0, fast=True):
    output = {}
    Higgs, tau1, tau2 = find_HTT(evt)
    if not Higgs :
        return output

    output["channel"], tau1, tau2, new_tau1, new_tau2 = determine_gen_channel(evt, tau1, tau2)

    store_vars.store_gen_ptc(evt, output, "Higgs", Higgs)
    store_vars.store_gen_ptc(evt, output, "tau1", tau1)
    store_vars.store_gen_ptc(evt, output, "tau2", tau2)

    leg1_pt, leg1_eta, leg1_phi = get_gen_leg_pt_eta_phi(evt, new_tau1, subchannel = output["channel"][0])
    leg2_pt, leg2_eta, leg2_phi = get_gen_leg_pt_eta_phi(evt, new_tau2, subchannel = output["channel"][1])

    output["leg1_pt"] = leg1_pt
    output["leg1_eta"] = leg1_eta
    output["leg1_phi"] = leg1_phi
    output["leg2_pt"] = leg2_pt
    output["leg2_eta"] = leg2_eta
    output["leg2_phi"] = leg2_phi

    if verbose > 2:
        print("\tHiggs energy is {} GeV.".format(Higgs.E))

    store_vars.store_gen_MET(evt, output)

    store_vars.store_evt_number(evt, output)
    
    return output
