import DL_for_HTT.HTT_analysis_Delphes.modules.store_vars as store_vars
from DL_for_HTT.HTT_analysis_Delphes.modules.utils import DR2
from math import sin, cos

def find_HTT(evt):
    ''' Find the Higgs boson in the generated particles,
    check that it decays in tau leptons,
    return the Higgs and the two taus.'''
    Higgs_IDs = [25, 35, 36]
    Higgs = [p for p in evt.Particle if abs(p.PID) in Higgs_IDs]
    evt_Higgs = {}
    for PID in Higgs_IDs:
        evt_Higgs[PID] = [p for p in Higgs if abs(p.PID) == PID]
    if 25 in evt_Higgs.keys() and (35 in evt_Higgs.keys() or 36 in evt_Higgs.keys()):
        # BSM Higgs decaying to SM Higgs, ignore event
        if any(evt.Particle.At(h.M1) in evt_Higgs[35]+evt_Higgs[36] for h in evt_Higgs[25]):
            return None, None, None
        elif any(evt.Particle.At(h.M2) in evt_Higgs[35]+evt_Higgs[36] for h in evt_Higgs[25]):
            return None, None, None
    for ptc in Higgs:
        if ptc.D1 != -1:
            if abs(evt.Particle.At(ptc.D1).PID) == 15:
                if ptc.D2 != -1:
                    if evt.Particle.At(ptc.D2).PID == - evt.Particle.At(ptc.D1).PID:
                        return ptc, evt.Particle.At(ptc.D1), evt.Particle.At(ptc.D2)
    return None, None, None

def determine_gen_channel(evt, tau1, tau2):
    leg1 = determine_tau_channel(evt, tau1)
    leg2 = determine_tau_channel(evt, tau1)
    channel = leg1+leg2
    # sort legs and channel
    if leg1 == leg2:
        if tau2.PT > tau1.PT:
            tau1, tau2 = tau2, tau1
    elif channel in ["tm", "te", "me"]:
        leg1, leg2 = leg2, leg1
        tau1, tau2 = tau2, tau1
    return leg1+leg2, tau1, tau2

def determine_tau_channel(evt, tau):
    decays_PIDs = [(tau, tau.PID)]

    while not any([abs(decays_PID[1]) in [12, 14, 16] for decays_PID in decays_PIDs]): # use neutrinos to check decays
        decays_idx = [x[0].D1 for x in decays_PIDs]
        decays_idx+= [x[0].D2 for x in decays_PIDs]
        decays_idx.append(-1)
        decays_idx = list(set(decays_idx))
        decays_idx.remove(-1)
        decays_PIDs = [(evt.Particle.At(x), evt.Particle.At(x).PID) for x in decays_idx]

    decays_PIDs = list(set(decays_PID[1] for decays_PID in decays_PIDs))

    if 12 in [abs(PID) for PID in decays_PIDs]:
        return "e"
    elif 14 in [abs(PID) for PID in decays_PIDs]:
        return "m"
    elif 16 in [abs(PID) for PID in decays_PIDs]:
        return "t"
    else:
        import pdb; pdb.set_trace()
    
def find_Higgs_production_final_state(evt, Higgs):
    #Higgs_mother = evt.Particle.At(Higgs.M1)
    Higgs_first = Higgs
    Higgs_mother1 = Higgs
    Higgs_mother2 = Higgs
    while Higgs_mother1.PID == Higgs.PID and Higgs_mother1 == Higgs_mother2:
        Higgs_first = Higgs_mother1
        Higgs_mother1 = evt.Particle.At(Higgs_first.M1)
        if Higgs_first.M2 != -1:
            Higgs_mother2 =evt.Particle.At(Higgs_first.M2)
        else:
            Higgs_mother2 = Higgs_mother1

    other_products = [p for p in evt.Particle if (p.M1 in [Higgs_first.M1, Higgs_first.M2] or p.M2 in [Higgs_first.M1, Higgs_first.M2]) and p != Higgs_first]
    return other_products

def find_jets_from(evt, ptc):
    return [j for j in evt.Jet if DR2(j, ptc)**.5 < 1]

def get_daughters(evt, ptc):
    ''' Sometime a gen ptc decays into a particle and itself (software feature),
    so this function tries to get the real decays.'''
    daughters = []
    if ptc.D1 != -1:
        D1 = evt.Particle.At(ptc.D1)
        if D1.PID == ptc.PID:
            daughters += get_daughters(evt, D1)
        else:
            daughters.append(D1)
    if ptc.D2 != -1:
        D2 = evt.Particle.At(ptc.D2)
        if D2.PID == ptc.PID:
            daughters += get_daughters(evt, D2)
        else:
            daughters.append(D2)
    return list(set(daughters))

def check_decays_from(evt, decays, ptc):
    final_decays = []
    correct_decays = set()
    bad_decays = set()
    for decay in decays:
        chain = []
        matched = False
        index = decay.M1
        while index != -1 and not matched:
            previous_decay = evt.Particle.At(index)
            if previous_decay in bad_decays:
                break
            chain.append(previous_decay)
            index = previous_decay.M1
            if previous_decay == ptc or previous_decay in correct_decays:
                matched = True
        if matched:
            final_decays.append(decay)
            correct_decays.update(chain)
        else:
            bad_decays.update(chain)
    return final_decays
    
def find_tau_decays(evt, tau):
    decays = []
    channel = "t"
    DM = None
    decays = check_decays_from(evt, evt.Particle, tau)
    decays = [p for p in set(decays) if p.PID != tau.PID]
    if any(abs(ptc.PID) == 11 for ptc in decays):
        channel = "e"
    elif any(abs(ptc.PID) == 13 for ptc in decays):
        channel = "m"
    if channel == "t":
        neutrinos = [neutrino for neutrino in decays if abs(neutrino.PID) in [12, 14, 16]]
        if len(neutrinos) > 1:
            string = "Hadronic tau with more than 1 neutrino at generator level??"
            #print(string)
            #import pdb; pdb.set_trace()
            #raise RuntimeError(string)
        pi0s = [pi0 for pi0 in decays if abs(pi0.PID) in [111]]
        photons = [p for p in decays if abs(p.PID) == 22]
        prongs = [p for p in decays if p.Charge != 0]
        DM = "{}prong{}pi0".format(len(prongs), len(pi0s))
        if not len(prongs) % 2:
            print(DM)
            #import pdb; pdb.set_trace()
    return decays, channel, DM
                        
def HTT_analysis(evt, verbose = 0, fast=True):
    output = {}
    Higgs, tau1, tau2 = find_HTT(evt)
    if not Higgs :
        return output

    output["channel"], tau1, tau2 = determine_gen_channel(evt, tau1, tau2)
    
    store_vars.store_gen_ptc(output, "Higgs", Higgs)
    store_vars.store_gen_ptc(output, "tau1", tau1)
    store_vars.store_gen_ptc(output, "tau2", tau2)

    if verbose > 2:
        print("\tHiggs energy is {} GeV.".format(Higgs.E))

    MET = evt.GenMissingET[0]
    output["MET_PT"] = MET.MET
    output["MET_Phi"] = MET.Phi


    #store_vars.store_evt_number(evt, output)

    if fast:
        return output
    
    other_products = find_Higgs_production_final_state(evt, Higgs)

    other_products.sort(key = lambda p : p.PT, reverse = True)
    jet1 = None
    jet2 = None
    if len(other_products) > 0:
        jet1_cands = find_jets_from(evt, other_products[0])
        if len(jet1_cands) > 1:
            jet1_cands.sort(key = lambda j : len(check_decays_from(evt, j.Particles, other_products[0])), reverse = True)
        if len(jet1_cands) > 0:
            jet1 = jet1_cands[0]
    if len(other_products) > 1:
        jet2_cands = find_jets_from(evt, other_products[1])
        if len(jet2_cands) > 1:
            jet2_cands.sort(key = lambda j : len(check_decays_from(evt, j.Particles, other_products[1])), reverse = True)
        if len(jet2_cands) > 0:
            jet2 = jet2_cands[0]
    if jet1 and jet2:
        if jet1.PT < jet2.PT:
            jet1, jet2 = jet2, jet1
    
    decays1, channel1, DM1 = find_tau_decays(evt, tau1)
    decays2, channel2, DM2 = find_tau_decays(evt, tau2)

    channel = channel1+channel2
    if channel in ["tt", "ee", "mm"]:
        if tau1.PT < tau2.PT:
            decays1, DM1, decays2, DM2 = decays2, DM2, decays1, DM1
            tau1, tau2 = tau2, tau1
    elif channel in ["me", "te", "tm"]:
        decays1, DM1, decays2, DM2 = decays2, DM2, decays1, DM1
        tau1, tau2 =tau2, tau1
        channel1, channel2 = channel2, channel1
        channel = channel1+channel2

    if verbose > 0:
        print("\tChannel: {}".format(channel))

    if verbose > 1:
        if any(DM != None for DM in [DM1, DM2]):
            DMstr = "\ttauh decay modes:"
            for DM in [DM1, DM2]:
                if DM is not None:
                    DMstr += " {}".format(DM)
            print(DMstr)

    if verbose >0:
        print("")
        print("\tleg1:")
        print("\ttau pT: {}, eta: {}, phi: {}, E: {}".format(tau1.PT, tau1.Eta, tau1.Phi, tau1.E))

        print("")
        print("\tleg2:")
        print("\ttau pT: {}, eta: {}, phi: {}, E: {}".format(tau2.PT, tau2.Eta, tau2.Phi, tau2.E))

    output = {
        "channel" : channel,
        "DM1" : DM1,
        "DM2" : DM2,
    }
    store_vars.store_jet(output, "jet1", jet1)
    store_vars.store_jet(output, "jet2", jet2)
    return output
