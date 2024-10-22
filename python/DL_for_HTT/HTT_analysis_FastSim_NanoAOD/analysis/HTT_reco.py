import DL_for_HTT.HTT_analysis_FastSim_NanoAOD.modules.store_vars as store_vars
import itertools
import numpy as np

import DL_for_HTT.common.HTT_cuts as common_cuts

def tauh_vs_jet_filter(evt, index, good_jets_list):
    eta = evt.GetLeaf("Tau_eta").GetValue(index)
    phi = evt.GetLeaf("Tau_phi").GetValue(index)
    for jet in good_jets_list:
        jet_eta = evt.GetLeaf("Jet_eta").GetValue(jet)
        jet_phi = evt.GetLeaf("Jet_phi").GetValue(jet)

        Delta_eta = abs(eta-jet_eta)
        Delta_phi = abs(phi-jet_phi)
        while Delta_phi > np.pi:
            Delta_phi -= 2*np.pi

        Delta_R2 = Delta_eta**2 + Delta_phi**2

        if Delta_R2 < 0.5**2:
            return False

    return True

def select_tauh_tt(evt, index, fakes=False):
    pT = evt.GetLeaf("Tau_pt").GetValue(index)
    eta = evt.GetLeaf("Tau_eta").GetValue(index)
    dz = evt.GetLeaf("Tau_dz").GetValue(index)
    charge = evt.GetLeaf("Tau_charge").GetValue(index)
    return all([
        pT > common_cuts.cut_tauh_tt_pt,
        abs(eta) < common_cuts.cut_tauh_tt_eta,
        abs(dz) < common_cuts.cut_tauh_tt_dz,
        abs(charge) == 1.,
        evt.GetLeaf("Tau_idDecayModeNewDMs").GetValue(index),
        evt.GetLeaf("Tau_decayMode").GetValue(index) in common_cuts.allowed_Tau_decayMode,
        evt.GetLeaf("Tau_idDeepTau2017v2p1VSe").GetValue(index) >= 2,
        evt.GetLeaf("Tau_idDeepTau2017v2p1VSmu").GetValue(index) >= 1,
        (evt.GetLeaf("Tau_idDeepTau2017v2p1VSjet").GetValue(index) >= 16 and not fakes) or (evt.GetLeaf("Tau_idDeepTau2017v2p1VSjet").GetValue(index) > 0 and evt.GetLeaf("Tau_idDeepTau2017v2p1VSjet").GetValue(index) < 16 and fakes),
        ])

def select_tauh_mt(evt, index, fakes=False):
    pT = evt.GetLeaf("Tau_pt").GetValue(index)
    eta = evt.GetLeaf("Tau_eta").GetValue(index)
    dz = evt.GetLeaf("Tau_dz").GetValue(index)
    charge = evt.GetLeaf("Tau_charge").GetValue(index)
    return all([
        pT >= common_cuts.cut_tauh_mt_pt,
        abs(eta) <= common_cuts.cut_tauh_mt_eta,
        abs(dz) < common_cuts.cut_tauh_mt_dz,
        abs(charge) == 1.,
        evt.GetLeaf("Tau_idDecayModeNewDMs").GetValue(index),
        evt.GetLeaf("Tau_decayMode").GetValue(index) in common_cuts.allowed_Tau_decayMode,
        evt.GetLeaf("Tau_idDeepTau2017v2p1VSe").GetValue(index) >= 2,
        evt.GetLeaf("Tau_idDeepTau2017v2p1VSmu").GetValue(index) >= 8,
        (evt.GetLeaf("Tau_idDeepTau2017v2p1VSjet").GetValue(index) >= 16 and not fakes) or (evt.GetLeaf("Tau_idDeepTau2017v2p1VSjet").GetValue(index) > 0 and evt.GetLeaf("Tau_idDeepTau2017v2p1VSjet").GetValue(index) < 16 and fakes),
        ])

def select_tauh_et(evt, index, fakes=False):
    pT = evt.GetLeaf("Tau_pt").GetValue(index)
    eta = evt.GetLeaf("Tau_eta").GetValue(index)
    dz = evt.GetLeaf("Tau_dz").GetValue(index)
    charge = evt.GetLeaf("Tau_charge").GetValue(index)
    return all([
        pT >= common_cuts.cut_tauh_et_pt,
        abs(eta) <= common_cuts.cut_tauh_et_eta,
        abs(dz) < common_cuts.cut_tauh_et_dz,
        abs(charge) == 1.,
        evt.GetLeaf("Tau_idDecayModeNewDMs").GetValue(index),
        evt.GetLeaf("Tau_decayMode").GetValue(index) in common_cuts.allowed_Tau_decayMode,
        evt.GetLeaf("Tau_idDeepTau2017v2p1VSe").GetValue(index) >= 32,
        evt.GetLeaf("Tau_idDeepTau2017v2p1VSmu").GetValue(index) >= 1,
        (evt.GetLeaf("Tau_idDeepTau2017v2p1VSjet").GetValue(index) >= 16 and not fakes) or (evt.GetLeaf("Tau_idDeepTau2017v2p1VSjet").GetValue(index) > 0 and evt.GetLeaf("Tau_idDeepTau2017v2p1VSjet").GetValue(index) < 16 and fakes),
        ])

def select_muon_mt(evt, index):
    pT = evt.GetLeaf("Muon_pt").GetValue(index)
    eta = evt.GetLeaf("Muon_eta").GetValue(index)
    dxy = evt.GetLeaf("Muon_dxy").GetValue(index)
    dz = evt.GetLeaf("Muon_dz").GetValue(index)
    return all([
        pT >= common_cuts.cut_muon_mt_pt,
        abs(eta) <= common_cuts.cut_muon_mt_eta,
        abs(dxy) < common_cuts.cut_muon_mt_d0,
        abs(dz) < common_cuts.cut_muon_mt_dz,
        evt.GetLeaf("Muon_mediumId").GetValue(index),
        evt.GetLeaf("Muon_pfRelIso04_all").GetValue(index) < 0.15,
        ])

def select_muon_em(evt, index):
    pT = evt.GetLeaf("Muon_pt").GetValue(index)
    eta = evt.GetLeaf("Muon_eta").GetValue(index)
    dxy = evt.GetLeaf("Muon_dxy").GetValue(index)
    dz = evt.GetLeaf("Muon_dz").GetValue(index)
    return all([
        pT > common_cuts.cut_muon_em_pt,
        abs(eta) < common_cuts.cut_muon_em_eta,
        abs(dxy) < common_cuts.cut_muon_em_d0,
        abs(dz) < common_cuts.cut_muon_em_dz,
        evt.GetLeaf("Muon_mediumId").GetValue(index),
        evt.GetLeaf("Muon_pfRelIso04_all").GetValue(index) < 0.15,
        ])

def select_muon_mm(evt, index):
    pT = evt.GetLeaf("Muon_pt").GetValue(index)
    eta = evt.GetLeaf("Muon_eta").GetValue(index)
    dxy = evt.GetLeaf("Muon_dxy").GetValue(index)
    dz = evt.GetLeaf("Muon_dz").GetValue(index)
    return all([
        pT > common_cuts.cut_muon_mm_pt,
        abs(eta) < common_cuts.cut_muon_mm_eta,
        abs(dxy) < common_cuts.cut_muon_mm_d0,
        abs(dz) < common_cuts.cut_muon_mm_dz,
        evt.GetLeaf("Muon_mediumId").GetValue(index),
        evt.GetLeaf("Muon_pfRelIso04_all").GetValue(index) < 0.15,
        ])

def select_muon_mt_dilepton_veto(evt, index):
    pT = evt.GetLeaf("Muon_pt").GetValue(index)
    eta = evt.GetLeaf("Muon_eta").GetValue(index)
    dxy = evt.GetLeaf("Muon_dxy").GetValue(index)
    dz = evt.GetLeaf("Muon_dz").GetValue(index)
    return all([
        pT > common_cuts.cut_muon_mt_dilepton_veto_pt,
        abs(eta) < common_cuts.cut_muon_mt_dilepton_veto_eta,
        evt.GetLeaf("Muon_looseId").GetValue(index),
        abs(dxy) < common_cuts.cut_muon_mt_dilepton_veto_d0,
        abs(dz) < common_cuts.cut_muon_mt_dilepton_veto_dz,
        evt.GetLeaf("Muon_pfRelIso04_all").GetValue(index) < common_cuts.cut_muon_mt_dilepton_veto_iso,
        ])

def select_muon_third_lepton_veto(evt, index):
    pT = evt.GetLeaf("Muon_pt").GetValue(index)
    eta = evt.GetLeaf("Muon_eta").GetValue(index)
    dxy = evt.GetLeaf("Muon_dxy").GetValue(index)
    dz = evt.GetLeaf("Muon_dz").GetValue(index)
    return all([
        pT > common_cuts.cut_muon_third_lepton_veto_pt,
        abs(eta) < common_cuts.cut_muon_third_lepton_veto_eta,
        evt.GetLeaf("Muon_mediumId").GetValue(index),
        abs(dxy) < common_cuts.cut_muon_third_lepton_veto_d0,
        abs(dz) < common_cuts.cut_muon_third_lepton_veto_dz,
        evt.GetLeaf("Muon_pfRelIso04_all").GetValue(index) < common_cuts.cut_muon_third_lepton_veto_iso,
        ])
        
def select_electron_et(evt, index):
    pT = evt.GetLeaf("Electron_pt").GetValue(index)
    eta = evt.GetLeaf("Electron_eta").GetValue(index)
    dxy = evt.GetLeaf("Electron_dxy").GetValue(index)
    dz = evt.GetLeaf("Electron_dz").GetValue(index)
    passConversionVeto = evt.GetLeaf("Electron_convVeto").GetValue(index)
    lostHits = evt.GetLeaf("Electron_lostHits").GetValue(index)
    mvaEleID_Fall17_noIso_V2_wp90 = evt.GetLeaf("Electron_mvaFall17V2noIso_WP90").GetValue(index)
    return all([
        pT >= common_cuts.cut_ele_et_pt,
        abs(eta) <= common_cuts.cut_ele_et_eta,
        abs(dxy) < common_cuts.cut_ele_et_d0,
        abs(dz) < common_cuts.cut_ele_et_dz,
        passConversionVeto,
        lostHits <= 1,
        mvaEleID_Fall17_noIso_V2_wp90,
        evt.GetLeaf("Electron_pfRelIso03_all").GetValue(index) < common_cuts.cut_ele_et_iso,
        ])

def select_electron_em(evt, index):
    pT = evt.GetLeaf("Electron_pt").GetValue(index)
    eta = evt.GetLeaf("Electron_eta").GetValue(index)
    dxy = evt.GetLeaf("Electron_dxy").GetValue(index)
    dz = evt.GetLeaf("Electron_dz").GetValue(index)
    passConversionVeto = evt.GetLeaf("Electron_convVeto").GetValue(index)
    lostHits = evt.GetLeaf("Electron_lostHits").GetValue(index)
    mvaEleID_Fall17_noIso_V2_wp90 = evt.GetLeaf("Electron_mvaFall17V2noIso_WP90").GetValue(index)
    return all([
            pT > common_cuts.cut_ele_em_pt,
            abs(eta) < common_cuts.cut_ele_em_eta,
            abs(dxy) < common_cuts.cut_ele_em_d0,
            abs(dz) < common_cuts.cut_ele_em_dz,
            passConversionVeto,
            lostHits <= 1,
            mvaEleID_Fall17_noIso_V2_wp90,
            evt.GetLeaf("Electron_pfRelIso03_all").GetValue(index) < common_cuts.cut_ele_em_iso,
        ])

def select_electron_ee(evt, index):
    pT = evt.GetLeaf("Electron_pt").GetValue(index)
    eta = evt.GetLeaf("Electron_eta").GetValue(index)
    dxy = evt.GetLeaf("Electron_dxy").GetValue(index)
    dz = evt.GetLeaf("Electron_dz").GetValue(index)
    passConversionVeto = evt.GetLeaf("Electron_convVeto").GetValue(index)
    lostHits = evt.GetLeaf("Electron_lostHits").GetValue(index)
    mvaEleID_Fall17_noIso_V2_wp90 = evt.GetLeaf("Electron_mvaFall17V2noIso_WP90").GetValue(index)
    return all([
            pT > common_cuts.cut_ele_ee_pt,
            abs(eta) < common_cuts.cut_ele_ee_eta,
            abs(dxy) < common_cuts.cut_ele_ee_d0,
            abs(dz) < common_cuts.cut_ele_ee_dz,
            passConversionVeto,
            lostHits <= 1,
            mvaEleID_Fall17_noIso_V2_wp90,
            evt.GetLeaf("Electron_pfRelIso03_all").GetValue(index) < common_cuts.cut_ele_ee_iso,
        ])

def select_electron_et_dilepton_veto(evt, index):
    pT = evt.GetLeaf("Electron_pt").GetValue(index)
    eta = evt.GetLeaf("Electron_eta").GetValue(index)
    dxy = evt.GetLeaf("Electron_dxy").GetValue(index)
    dz = evt.GetLeaf("Electron_dz").GetValue(index)
    passConversionVeto = evt.GetLeaf("Electron_convVeto").GetValue(index)
    lostHits = evt.GetLeaf("Electron_lostHits").GetValue(index)
    return all([
        pT > common_cuts.cut_ele_et_dilepton_veto_pt,
        abs(eta) < common_cuts.cut_ele_et_dilepton_veto_eta,
        abs(dxy) < common_cuts.cut_ele_et_dilepton_veto_d0,
        abs(dz) < common_cuts.cut_ele_et_dilepton_veto_dz,
        passConversionVeto,
        lostHits <= 1,
        #ele.id_passes('cutBasedElectronID-Fall17-94X-V2', 'veto'),
        evt.GetLeaf("Electron_pfRelIso03_all").GetValue(index) < common_cuts.cut_ele_et_dilepton_veto_iso,
        ])

def select_electron_third_lepton_veto(evt, index):
    pT = evt.GetLeaf("Electron_pt").GetValue(index)
    eta = evt.GetLeaf("Electron_eta").GetValue(index)
    dxy = evt.GetLeaf("Electron_dxy").GetValue(index)
    dz = evt.GetLeaf("Electron_dz").GetValue(index)
    passConversionVeto = evt.GetLeaf("Electron_convVeto").GetValue(index)
    lostHits = evt.GetLeaf("Electron_lostHits").GetValue(index)
    mvaEleID_Fall17_noIso_V2_wp90 = evt.GetLeaf("Electron_mvaFall17V2noIso_WP90").GetValue(index)
    return all([
        pT > common_cuts.cut_ele_third_lepton_veto_pt,
        abs(eta) < common_cuts.cut_ele_third_lepton_veto_eta,
        abs(dxy) < common_cuts.cut_ele_third_lepton_veto_d0,
        abs(dz) < common_cuts.cut_ele_third_lepton_veto_dz,
        passConversionVeto,
        lostHits <= 1,
        mvaEleID_Fall17_noIso_V2_wp90,
        evt.GetLeaf("Electron_pfRelIso03_all").GetValue(index) < common_cuts.cut_ele_third_lepton_veto_iso,
    ])
    
def select_tauh(evt, tau_idx, channel, good_jets_list, fakes=False):
    jet_clean = tauh_vs_jet_filter(evt, tau_idx, good_jets_list)
    if not jet_clean and not fakes:
        return False
    if channel == "tt":
        return select_tauh_tt(evt, tau_idx, fakes)
    elif channel == "mt":
        return select_tauh_mt(evt, tau_idx, fakes)
    elif channel == "et":
        return select_tauh_et(evt, tau_idx, fakes)

def select_muon(evt, muon_idx, channel):
    if channel == "mt":
        return select_muon_mt(evt, muon_idx)
    elif channel == "mm":
        return select_muon_mm(evt, muon_idx)
    elif channel == "em":
        return select_muon_em(evt, muon_idx)

def select_electron(evt, ele_idx, channel):
    if channel == "et":
        return select_electron_et(evt, ele_idx)
    elif channel == "ee":
        return select_electron_ee(evt, ele_idx)
    elif channel == "em":
        return select_electron_em(evt, ele_idx)

def select_jet_20(evt, index):
    pT = evt.GetLeaf("Jet_pt").GetValue(index)
    eta = evt.GetLeaf("Jet_eta").GetValue(index)
    return all([
        pT > 20,
        abs(eta) < 4.7,
    ])

def select_jet_30(evt, index):
    pT = evt.GetLeaf("Jet_pt").GetValue(index)
    eta = evt.GetLeaf("Jet_eta").GetValue(index)
    return all([
        pT > 30,
        abs(eta) < 4.7,
    ])

def select_jet_B(evt, index):
    pT = evt.GetLeaf("Jet_pt").GetValue(index)
    eta = evt.GetLeaf("Jet_eta").GetValue(index)
    deepB = evt.GetLeaf("Jet_btagDeepB").GetValue(index)
    return all([
        pT > 20,
        abs(eta) < 2.5,
        deepB > 0.3033,
    ])

names_to_letter = {
    "Electrons" : "e",
    "Muons" : "m",
    "Taus" : "t",
}

letter_to_names = {}
for k, v  in names_to_letter.iteritems():
    letter_to_names[v] = k

def find_tau_DM(tau):
    tracks = [c for c in tau.Constituents if hasattr(c, 'Charge')]
    Nprongs = len(tracks)
    if not Nprongs %2:
        Nprongs -= 1
    towers = [c for c in tau.Constituents if not hasattr(c, 'Charge')]
    if len(towers) > 1:
        Npi0s = int(len(towers)/2)
    else:
        Npi0s = len(towers)
    DM = "{}prong{}pi0".format(Nprongs, Npi0s)
    return DM

def HTT_analysis(evt, accepted_channels = ["tt", "mt", "et", "mm", "ee", "em"], verbose = 0, cutflow_stats = {}, fakes=False):

    if not all([
            evt.GetLeaf("Flag_goodVertices").GetValue(0),
            evt.GetLeaf("Flag_globalSuperTightHalo2016Filter").GetValue(0),
            evt.GetLeaf("Flag_HBHENoiseFilter").GetValue(0),
            evt.GetLeaf("Flag_HBHENoiseIsoFilter").GetValue(0),
            evt.GetLeaf("Flag_EcalDeadCellTriggerPrimitiveFilter").GetValue(0),
            evt.GetLeaf("Flag_BadPFMuonFilter").GetValue(0),
            evt.GetLeaf("Flag_eeBadScFilter").GetValue(0),
            evt.GetLeaf("Flag_ecalBadCalibFilter").GetValue(0),
    ]):
        return {}, cutflow_stats
    
    # retreive objects for event
    nElectron = int(evt.GetLeaf("nElectron").GetValue(0))
    nMuon = int(evt.GetLeaf("nMuon").GetValue(0))
    nTau = int(evt.GetLeaf("nTau").GetValue(0))
    nJet = int(evt.GetLeaf("nJet").GetValue(0))

    bad_jets_list = []
    for tau in range(nTau):
        bad_jets_list.append(evt.GetLeaf("Tau_jetIdx").GetValue(tau))

    # tauh cleaning
    good_jets_list = [idx for idx in range(nJet) if not idx in bad_jets_list]

    # jet ID tight lepton veto
    good_jets_list = [idx for idx in good_jets_list if evt.GetLeaf("Jet_jetId").GetValue(idx) >= 6]

    # jet 20
    jets20_list = [idx for idx in good_jets_list if select_jet_20(evt, idx)]

    # b jets
    bjets_list = [idx for idx in good_jets_list if select_jet_B(evt, idx)]
    jets30_list = [idx for idx in good_jets_list if select_jet_30(evt, idx)]

    good_jets_list = jets30_list+bjets_list

    # select leptons for accepted channels
    selections = {}
    possible_channels = [c for c in accepted_channels]
    for channel in accepted_channels:
        min_to_found = 2 if channel[0] == channel[1] else 1
        selections[channel] = {}
        if "t" in channel:
            selections[channel]["Taus"] = [tau for tau in range(nTau) if select_tauh(evt, tau, channel, good_jets_list, fakes)]
            if len(selections[channel]["Taus"]) < min_to_found:
                possible_channels.remove(channel)
                continue
        if "m" in channel:
            selections[channel]["Muons"] = [muon for muon in range(nMuon) if select_muon(evt, muon, channel)]
            if len(selections[channel]["Muons"]) < min_to_found:
                possible_channels.remove(channel)
                continue
        if "e" in channel:
            selections[channel]["Electrons"] = [ele for ele in range(nElectron) if select_electron(evt, ele, channel)]
            if len(selections[channel]["Electrons"]) < min_to_found:
                possible_channels.remove(channel)
                continue

    if len(possible_channels) == 0:
        return {}, cutflow_stats

    # construc dilepton
    dilepton = {}
    for channel in [c for c in accepted_channels if c in possible_channels]:
        if channel in ["em", "mm"]:
            DRmin = .3
        else:
            DRmin = .5
        l1s = selections[channel][letter_to_names[channel[0]]]
        l2s = selections[channel][letter_to_names[channel[1]]]
        pairs = []
        if channel[0] == channel[1]:
            for l1, l2 in itertools.combinations(l1s,2):
                pairs.append((l1,l2))
        else:
            for l1 in l1s:
                for l2 in l2s:
                    pairs.append((l1,l2))
        # check opposite sign
        pairs_OS = []
        for pair in pairs:
            c1 = evt.GetLeaf("{}_charge".format(letter_to_names[channel[0]][:-1])).GetValue(pair[0])
            c2 = evt.GetLeaf("{}_charge".format(letter_to_names[channel[1]][:-1])).GetValue(pair[1])
            if c1 == -c2:
                pairs_OS.append(pair)
        # check DR2
        pairs_OS_DR2 = []
        for pair in pairs:
            eta1 = evt.GetLeaf("{}_eta".format(letter_to_names[channel[0]][:-1])).GetValue(pair[0])
            eta2 = evt.GetLeaf("{}_eta".format(letter_to_names[channel[1]][:-1])).GetValue(pair[1])
            phi1 = evt.GetLeaf("{}_phi".format(letter_to_names[channel[0]][:-1])).GetValue(pair[0])
            phi2 = evt.GetLeaf("{}_phi".format(letter_to_names[channel[1]][:-1])).GetValue(pair[1])
            DR2 = (eta1-eta2)**2 + (phi1-phi2)**2
            if DR2 > DRmin**2:
                pairs_OS_DR2.append(pair)
        pairs = pairs_OS_DR2
        if len(pairs) == 0:
            possible_channels.remove(channel)
            continue
        metric = lambda dl: (-evt.GetLeaf("{}_pt".format(letter_to_names[channel[0]][:-1])).GetValue(pair[0]), -evt.GetLeaf("{}_pt".format(letter_to_names[channel[1]][:-1])).GetValue(pair[1]))
        pairs = sorted(pairs, key=metric, reverse=False)
        dilepton[channel] = pairs[0]
        if verbose >0:
            tau1, tau2 = dilepton[channel]
            print(channel)

    # Lepton vetoes
    muon_third_lepton = [muon for muon in range(nMuon) if select_muon_third_lepton_veto(evt, muon)]
    for channel in ["tt", "et", "ee"]:
        if channel in possible_channels and len(muon_third_lepton)>0:
            possible_channels.remove(channel)
    for channel in ["mt", "em"]:
        if channel in possible_channels and len(muon_third_lepton)>1:
            possible_channels.remove(channel)
    for channel in ["mm"]:
        if channel in possible_channels and len(muon_third_lepton)>2:
            possible_channels.remove(channel)
    ele_third_lepton = [ele for ele in range(nElectron) if select_electron_third_lepton_veto(evt, ele)]
    for channel in ["tt", "mt", "mm"]:
        if channel in possible_channels and len(ele_third_lepton)>0:
            possible_channels.remove(channel)
    for channel in ["et", "em"]:
        if channel in possible_channels and len(ele_third_lepton)>1:
            possible_channels.remove(channel)
    for channel in ["ee"]:
        if channel in possible_channels and len(ele_third_lepton)>2:
            possible_channels.remove(channel)

    if "mt" in possible_channels:
        rm_channel = False
        dimu_veto = [muon for muon in range(nMuon) if select_muon_mt_dilepton_veto(evt, muon) and not muon == dilepton["mt"][0]]
        for mu1, mu2 in itertools.combinations(dimu_veto,2):
            c1 = evt.GetLeaf("Muon_charge").GetValue(mu1)
            c2 = evt.GetLeaf("Muon_charge").GetValue(mu2)
            eta1 = evt.GetLeaf("Muon_charge").GetValue(mu1)
            eta2 = evt.GetLeaf("Muon_charge").GetValue(mu2)
            phi1 = evt.GetLeaf("Muon_charge").GetValue(mu1)
            phi2 = evt.GetLeaf("Muon_charge").GetValue(mu2)
            if c1 * c2 < 0 and ((eta1-eta2)**2 + (phi1-phi2)**2)**.5 > 0.15:
                rm_channel = True
        if rm_channel:
            possible_channels.remove("mt")

    if "et" in possible_channels:
        rm_channel = False
        diele_vetoe = [ele for ele in range(nElectron) if select_electron_et_dilepton_veto(evt, ele) and not ele == dilepton["et"][0]]
        for e1, e2 in itertools.combinations(diele_vetoe,2):
            c1 = evt.GetLeaf("Electron_charge").GetValue(e1)
            c2 = evt.GetLeaf("Electron_charge").GetValue(e2)
            eta1 = evt.GetLeaf("Electron_charge").GetValue(e1)
            eta2 = evt.GetLeaf("Electron_charge").GetValue(e2)
            phi1 = evt.GetLeaf("Electron_charge").GetValue(e1)
            phi2 = evt.GetLeaf("Electron_charge").GetValue(e2)
            if c1 * c2 < 0 and ((eta1-eta2)**2 + (phi1-phi2)**2)**.5 > 0.15:
                rm_channel = True
        if rm_channel:
            possible_channels.remove("et")

    if len(possible_channels) > 1:
        raise RuntimeError("More than one channel is still possible, please check!")
    elif len(possible_channels) == 0:
        return {}, cutflow_stats
        
    channel = possible_channels[0]
    tau1, tau2 = dilepton[channel]


    # store variables
    output = {
        "channel" : channel,
    }

    # legs
    store_vars.store_HTT_leg(evt, output, "tau1", tau1, type=channel[0])
    store_vars.store_HTT_leg(evt, output, "tau2", tau2, type=channel[1])
    
    # MET and METcov
    store_vars.store_reco_MET(evt, output)
    
    # PU primary vertices
    store_vars.store_reco_PU(evt, output)

    # Up to two leading jets
    store_vars.store_none(output, "jet1", type="jet")
    store_vars.store_none(output, "jet2", type="jet")
    for k in range(int(min([2, len(good_jets_list)]))):
        store_vars.store_jet(evt, output, "jet{}".format(k+1), good_jets_list[k])

    # Up to two leading b-jets
    store_vars.store_none(output, "bjet1", type="jet")
    store_vars.store_none(output, "bjet2", type="jet")
    for k in range(int(min([2, len(bjets_list)]))):
        store_vars.store_jet(evt, output, "bjet{}".format(k+1), bjets_list[k])
    output["Nbjets"] = len(bjets_list)

    # Remaining_Jets (other jets) computation and storage
    remaining_jets_px = 0
    remaining_jets_py = 0
    remaining_jets_pz = 0
    N_jets = 0
    if len(good_jets_list) > 2:
        for k in range(2, len(good_jets_list)):
            jet_idx = good_jets_list[k]
            N_jets+=1
            remaining_jets_px += evt.GetLeaf("Jet_pt").GetValue(jet_idx) * np.cos(evt.GetLeaf("Jet_phi").GetValue(jet_idx))
            remaining_jets_py += evt.GetLeaf("Jet_pt").GetValue(jet_idx) * np.sin(evt.GetLeaf("Jet_phi").GetValue(jet_idx))
            remaining_jets_pz += evt.GetLeaf("Jet_pt").GetValue(jet_idx) * np.sinh(evt.GetLeaf("Jet_eta").GetValue(jet_idx))
    if N_jets == 0:
        remaining_jets_pt = 0
        remaining_jets_phi = 0
        remaining_jets_eta = 0
    else:
        remaining_jets_pt = np.sqrt(remaining_jets_px**2 + remaining_jets_py**2)
        remaining_jets_phi = np.arcsin(remaining_jets_py/remaining_jets_pt)
        remaining_jets_eta = np.arccosh(np.sqrt(remaining_jets_px**2 + remaining_jets_py**2 + remaining_jets_pz**2) / remaining_jets_pt) * np.sign(remaining_jets_pz)
    store_vars.store_remaining_jets(evt, output, "remaining_jets", remaining_jets_pt, remaining_jets_eta, remaining_jets_phi, N_jets)
        
    return output, cutflow_stats
