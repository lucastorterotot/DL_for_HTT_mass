import Delphes_to_NN.modules.store_vars as store_vars
from Delphes_to_NN.modules.utils import DR2, get_MET_and_METcov
import itertools

def select_tauh_tt(tau):
    return all([
        tau.PT > 40,
        abs(tau.Eta) < 2.1,
        #abs(tau.dz) < 0.2,
        #tau.tauID('decayModeFinding') > 0.5
        abs(tau.Charge) == 1.,
        #tau.tauID('byVVLooseIsolationMVArun2017v2DBoldDMwLT2017')
        ])

def select_tauh_mt(tau):
    return all([
        tau.PT >= 23,
        abs(tau.Eta) <= 2.3,
        #abs(tau.dz) < 0.2,
        #tau.tauID('decayModeFinding') > 0.5,
        abs(tau.Charge) == 1.,
        #tau.tauID('byVVLooseIsolationMVArun2017v2DBoldDMwLT2017'),
        ])

def select_tauh_et(tau):
    return select_tauh_mt(tau)

def select_muon_mt(muon):
    return all([
        muon.PT >= 21,
        abs(muon.Eta) <= 2.1,
        #abs(muon.dxy) < 0.045,
        #abs(muon.dz) < 0.2,
        #muon.isMediumMuon,
        ])

def select_muon_em(muon):
    return all([
        muon.PT > 13,
        abs(muon.Eta) < 2.4,
        #abs(muon.dxy) < 0.045,
        #abs(muon.dz) < 0.2,
        #muon.isMediumMuon,
        ])

def select_muon_mt_dilepton_veto(muon):
    return all([
        muon.PT > 15,
        abs(muon.Eta) < 2.4,
        #muon.isLooseLMuon,
        #abs(muon.dxy) < 0.045,
        #abs(muon.dz) < 0.2,
        #muon.iso_htt < 0.3,
        ])

def select_muon_third_lepton_veto(muon):
    return all([
        muon.PT > 10,
        abs(muon.Eta) < 2.4,
        #muon.isMediumLMuon,
        #abs(muon.dxy) < 0.045,
        #abs(muon.dz) < 0.2,
        #muon.iso_htt < 0.3,
        ])
        
def select_electron_et(ele):
    return all([
        ele.PT >= 25,
        abs(ele.Eta) <= 2.1,
        #abs(ele.dxy) < 0.045,
        #abs(ele.dz) < 0.2,
        #ele.passConversionVeto,
        #ele.gsfTrack().hitPattern().numberOfLostHits(ROOT.reco.HitPattern.MISSING_INNER_HITS) <= 1,
        #ele.id_passes("mvaEleID-Fall17-noIso-V2","wp90"),
        ])

def select_electron_em(ele):
    return all([
            ele.PT > 13,
            abs(ele.Eta) < 2.5,
            #abs(ele.dxy) < 0.045,
            #abs(ele.dz) < 0.2,
            #ele.passConversionVeto,
            #ele.gsfTrack().hitPattern().numberOfLostHits(ROOT.reco.HitPattern.MISSING_INNER_HITS) <= 1,
            #ele.id_passes("mvaEleID-Fall17-noIso-V2","wp90"),
        ])

def select_electron_et_dilepton_veto(ele):
    return all([
        ele.PT > 15,
        abs(ele.Eta) < 2.5,
        #abs(ele.dxy) < 0.045,
        #abs(ele.dz) < 0.2,
        #ele.passConversionVeto,
        #ele.gsfTrack().hitPattern().numberOfLostHits(ROOT.reco.HitPattern.MISSING_INNER_HITS) <= 1,
        #ele.id_passes('cutBasedElectronID-Fall17-94X-V2', 'veto'),
        #ele.iso_htt < 0.3,
        ])

def select_electron_third_lepton_veto(ele):
    return all([
        ele.PT > 10,
        abs(ele.Eta) < 2.5,
        #abs(ele.dxy) < 0.045,
        #abs(ele.dz) < 0.2,
        #ele.passConversionVeto,
        #ele.gsfTrack().hitPattern().numberOfLostHits(ROOT.reco.HitPattern.MISSING_INNER_HITS) <= 1,
        #ele.id_passes("mvaEleID-Fall17-noIso-V2","wp90"),
        #ele.iso_htt < 0.3,
    ])
    
def select_tauh(tau, channel):
    if channel == "tt":
        return select_tauh_tt(tau)
    elif channel == "mt":
        return select_tauh_mt(tau)
    elif channel == "et":
        return select_tauh_et(tau)

def select_muon(muon, channel):
    if channel == "mt":
        return select_muon_mt(muon)
    elif channel == "mm":
        return False # select_muon_mm(muon)
    elif channel == "em":
        return select_muon_em(muon)

def select_electron(ele, channel):
    if channel == "et":
        return select_electron_et(ele)
    elif channel == "ee":
        return False # select_electron_ee(ele)
    elif channel == "em":
        return select_electron_em(ele)


names_to_letter = {
    "electrons" : "e",
    "muons" : "m",
    "taus" : "t",
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

def HTT_analysis(evt, accepted_channels = ["tt", "mt", "et", "mm", "ee", "em"], verbose = 0):
    # retreive objects for event
    
    MET = evt.MissingET
    photons = [p for p in evt.Photon]
    electrons = [e for e in evt.Electron]
    muons = [m for m in evt.Muon]
    jets = [j for j in evt.Jet if not j.TauTag]
    taus = [j for j in evt.Jet if j.TauTag]

    # select leptons for accepted channels
    selections = {}
    possible_channels = [c for c in accepted_channels]
    for channel in accepted_channels:
        min_to_found = 2 if channel[0] == channel[1] else 1
        selections[channel] = {}
        if "t" in channel:
            selections[channel]["taus"] = [tau for tau in taus if select_tauh(tau, channel)]
            if len(selections[channel]["taus"]) < min_to_found:
                possible_channels.remove(channel)
                continue
        if "m" in channel:
            selections[channel]["muons"] = [muon for muon in muons if select_muon(muon, channel)]
            if len(selections[channel]["muons"]) < min_to_found:
                possible_channels.remove(channel)
                continue
        if "e" in channel:
            selections[channel]["electrons"] = [ele for ele in electrons if select_electron(ele, channel)]
            if len(selections[channel]["electrons"]) < min_to_found:
                possible_channels.remove(channel)
                continue

    if len(possible_channels) == 0:
        return {}

    # construc dilepton
    dilepton = {}
    for channel in [c for c in accepted_channels if c in possible_channels]:
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
        _pairs = [p for p in pairs]
        pairs = [pair for pair in pairs if pair[0].Charge == - pair[1].Charge]
        pairs = [pair for pair in pairs if DR2(pair[0], pair[1])**.5 > .5] # check
        if len(pairs) == 0:
            possible_channels.remove(channel)
            continue
        metric = lambda dl: (-dl[0].PT, -dl[1].PT)
        pairs = sorted(pairs, key=metric, reverse=False)
        dilepton[channel] = pairs[0]
        if verbose >0:
            tau1, tau2 = dilepton[channel]
            print(channel)
            print("")
            print("\tleg1:")
            print("\ttau pT: {}, eta: {}, phi: {}".format(tau1.PT, tau1.Eta, tau1.Phi))

            print("")
            print("\tleg2:")
            print("\ttau pT: {}, eta: {}, phi: {}".format(tau2.PT, tau2.Eta, tau2.Phi))
                                                        

    # select only one channel
    if len(possible_channels) > 1:
        muon_third_lepton = [m for m in muons if select_muon_third_lepton_veto(m)]
        ele_third_lepton = [e for e in electrons if select_electron_third_lepton_veto(e)]
        for channel in [c for c in accepted_channels if c in possible_channels]:
            if channel == "mt":
                muons_dilepton = [m for m in muons if select_muon_mt_dilepton_veto(m) if not m in dilepton[channel]]
                if len(muons_dilepton) > len(selections[channel]["muons"]) or len(ele_third_lepton)>0:
                    possible_channels.remove(channel)
            if channel == "et":
                ele_dilepton = [e for e in electrons if select_electron_et_dilepton_veto(e) if not e in dilepton[channel]]
                if len(ele_dilepton) > len(selections[channel]["electrons"]) or len(muon_third_lepton)>0:
                    possible_channels.remove(channel)
            if channel == "tt":
                if len(ele_third_lepton)>0 or len(muon_third_lepton)>0:
                    possible_channels.remove(channel)

    if len(possible_channels) > 1:
        raise RuntimeError("More than one channel is still possible, please check!")
    elif len(possible_channels) == 0:
        return {}
        
    channel = possible_channels[0]
    tau1, tau2 = dilepton[channel]
    decays1, decays2 = tau1, tau2
    DM1, DM2 = None, None

    if channel[0] == "t":
        DM1 = find_tau_DM(tau1)
    if channel[1] == "t":
        DM2 = find_tau_DM(tau2)


    # MET and METcov
    MET, METcov = get_MET_and_METcov(evt)

    # Store two leading jets
    jets.sort(key = lambda j : j.PT, reverse = True)

    jet1 = None
    jet2 = None
    if len(jets) > 0:
        jet1 = jets[0]
    if len(jets) > 1:
        jet2 = jets[1]
        
    output = {
        "channel" : channel,
        "DM1" : DM1,
        "DM2" : DM2,
        "MET_PT" : MET.MET,
        "MET_Phi" : MET.Phi,
        "METcov_xx" : METcov[0][0],
        "METcov_xy" : METcov[0][1],
        "METcov_yy" : METcov[1][1],
    }
    store_vars.store_real_tau_decays(output, "tau1", tau1, type=channel[0])
    store_vars.store_real_tau_decays(output, "tau2", tau2, type=channel[1])
    store_vars.store_jet(output, "jet1", jet1)
    store_vars.store_jet(output, "jet2", jet2)
    return output
