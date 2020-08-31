from math import cos, sin, sqrt

def DR2(ptc1, ptc2):
    return (ptc1.Eta - ptc2.Eta)**2 + (ptc1.Phi - ptc2.Phi)**2

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

def cleanJet(jet, leptons, dRMatch = 0.4):
    for lep_i in leptons:
        if DR2(jet,lep_i) < dRMatch**2:
            return False
    return True
    
def get_MET_and_METcov(evt):
    jets = list(evt.Jet) # consider taus in jets
    leptons = list(evt.Electron)+list(evt.Muon)
    photons = list(evt.Photon)

    pfCandidates = jets+leptons+photons
    
    cov_xx = 0
    cov_xy = 0
    cov_yy = 0

    # subtract leptons out of sumPtUnclustered
    footprint = set()
    for lep_i in leptons:
        if lep_i.PT > 10:
            footprint.add(lep_i)

    # subtract jets out of sumPtUnclustered
    for jet in jets:
        if not cleanJet(jet, leptons):
            continue
        footprint.add(jet)

    # calculate sumPtUnclustered
    sumPtUnclustered = 0
    for ptc in pfCandidates:
        if ptc not in footprint:
            sumPtUnclustered += ptc.PT

    # add jets to metsig covariance matrix and subtract them from sumPtUnclustered
    for jet in jets:
        if not cleanJet(jet, leptons):
            continue

        jpt = jet.PT
        
        if jpt > 15:
            jeta = jet.Eta
            feta = abs(jet.Eta)
            jphi = jet.Phi
            c = cos(jphi)
            s = sin(jphi)
            
            dpt = get_sigma_et(jet)
            dph = get_sigma_tan(jet)
            
            cov_xx += dpt * dpt * c * c + dph * dph * s * s
            cov_xy += (dpt * dpt - dph * dph) * c * s
            cov_yy += dph * dph * c * c + dpt * dpt * s * s
            
        else:
            sumPtUnclustered += jpt
        
    # add pseudo-jet to metsig covariance matrix
    pjetParams_ = (-0.2586,0.6173)
    cov_xx += pjetParams_[0] * pjetParams_[0] + pjetParams_[1] * pjetParams_[1] * sumPtUnclustered
    cov_yy += pjetParams_[0] * pjetParams_[0] + pjetParams_[1] * pjetParams_[1] * sumPtUnclustered

    return evt.MissingET[0], [[cov_xx, cov_xy], [cov_xy, cov_yy]]
