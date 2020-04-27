from Delphes_to_NN.modules.utils import DR2
import itertools
from math import cos, sin, sqrt

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

def get_sigma_et(ptc):
    return get_MET_resolution(ptc, "ET")

def get_sigma_tan(ptc):
    return get_MET_resolution(ptc, "PHI")

def get_MET_resolution(ptc, type):
    et = ptc.PT
    if type == "ET":
        par = (0.05, 0, 0)
        return et * sqrt((par[2] * par[2]) + (par[1] * par[1] / et) + (par[0] * par[0] / (et * et)))
    if type == "PHI":
        par = 0.002
        return par * et
    
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
    MET = evt.MissingET[0]
    METcov = [[0, 0], [0, 0]]

    xmet_ = 0#MET.MET * cos(MET.Phi)
    ymet_ = 0#MET.MET * sin(MET.Phi)

    # photons = [p for p in evt.Photon]
    # electrons = [e for e in evt.Electron]
    # muons = [m for m in evt.Muon]
    # jets = [j for j in evt.Jet if not j.TauTag]
    # taus = [j for j in evt.Jet if j.TauTag]
    for ptc in list(evt.Photon)+list(evt.Electron)+list(evt.Muon)+list(evt.Jet):
        et_tmp = ptc.PT
        phi_tmp = ptc.Phi
        sigma_et = get_sigma_et(ptc)
        sigma_tan = get_sigma_tan(ptc)
        cosphi = cos(phi_tmp)
        sinphi = sin(phi_tmp)
        
        xmet_ -= et_tmp * cosphi
        ymet_ -= et_tmp * sinphi

        sigma0_2 = sigma_et * sigma_et
        sigma1_2 = sigma_tan * sigma_tan

        METcov[0][0] += sigma0_2 * cosphi * cosphi + sigma1_2 * sinphi * sinphi
        METcov[0][1] += cosphi * sinphi * (sigma0_2 - sigma1_2)
        METcov[1][0] += cosphi * sinphi * (sigma0_2 - sigma1_2)
        METcov[1][1] += sigma1_2 * cosphi * cosphi + sigma0_2 * sinphi * sinphi
        
    output = {
        "channel" : channel,
        "leg1" : (tau1, decays1, DM1),
        "leg2" : (tau2, decays2, DM2),
        "MET" : MET,
        "METcov" : METcov,
    }

    return output
