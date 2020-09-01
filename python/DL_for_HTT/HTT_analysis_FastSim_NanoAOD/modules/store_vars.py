def store_evt_number(evt, dic):
    dic["Event"] = evt.GetLeaf("event").GetValue()

default_attrs = ["pt", "eta", "phi"]
def store(evt, dic, name, index, branch_prefix, attrs = default_attrs):
    for attr in attrs:
        if attr == "pdgId" and branch_prefix == "Tau":
            dic["{}_{}".format(name, "pdgId")] = evt.GetLeaf("{}_{}".format(branch_prefix, "charge")).GetValue(index) * -15
        else:
            dic["{}_{}".format(name, attr)] = evt.GetLeaf("{}_{}".format(branch_prefix, attr)).GetValue(index)
    if "pdgId" in attrs:
        # GenPart_mass :
        # Mass stored for all particles with the exception of quarks (except top), leptons/neutrinos, photons with mass < 1 GeV, gluons, pi0(111), pi+(211), D0(421), and D+(411). For these particles, you can lookup the value from PDG.
        # Values are taken from PDG booklet 2018
        if  abs(dic["{}_{}".format(name, "pdgId")]) == 15: # tau
            dic["{}_{}".format(name, "mass")] = 1.77686
        elif abs(dic["{}_{}".format(name, "pdgId")])== 13: # muon
            dic["{}_{}".format(name, "mass")] = 0.1056583745
        elif abs(dic["{}_{}".format(name, "pdgId")])== 11: # electron
            dic["{}_{}".format(name, "mass")] = 510.9989461 * 10**(-6)


default_gen_ptc_attrs = ["pdgId", "mass", "pt", "eta", "phi"]
def store_gen_ptc(evt, dic, name, gen_ptc_idx, attrs = default_gen_ptc_attrs):
    store(evt, dic, name, gen_ptc_idx, "GenPart", attrs = attrs)

def store_gen_MET(evt, dic):
    store(evt, dic, "MET", 0, "GenMET",
          attrs = ["pt", "phi"])

def store_reco_MET(evt, dic):
    store(evt, dic, "MET", 0, "MET",
          attrs = ["pt", "phi", "covXX", "covXY", "covYY", "significance"])

default_jet_attrs = default_attrs + ["btagDeepB"]
def store_jet(evt, dic, name, jet_index, attrs = default_jet_attrs):
    store(evt, dic, name, jet_index, "Jet", attrs = attrs)

def store_vars.store_recoil(evt, dic, name, recoil_pt, recoil_pz):
    dic["{}_{}".format(name, "pt")] = recoil_pt
    dic["{}_{}".format(name, "pz")] = recoil_pz

default_HTT_leg_attrs = default_attrs + ["charge", "pdgId"]
def store_HTT_leg(evt, dic, name, ptc_index, type=None, attrs = default_HTT_leg_attrs):
    if type == "t":
        store_tauh(evt, dic, name, ptc_index, attrs=attrs)
    elif type == "m":
        store_muon(evt, dic, name, ptc_index, attrs=attrs)
    elif type =="e":
        store_electron(evt, dic, name, ptc_index, attrs=attrs)

def store_tauh(evt, dic, name, tauh_index,  attrs = default_HTT_leg_attrs):
    store(evt, dic, name, tauh_index, "Tau", attrs = attrs)

def store_muon(evt, dic, name, muon_index, attrs = default_HTT_leg_attrs):
    store(evt, dic, name, muon_index, "Muon", attrs = attrs)

def store_electron(evt, dic, name, ele_index, attrs = default_HTT_leg_attrs):
    store(evt, dic, name, ele_index, "Electron", attrs = attrs)

def store_none(dic, name, type="jet"):
    attrs = []
    if type == "jet":
        attrs = default_jet_attrs
    for attr in attrs:
        dic["{}_{}".format(name, attr)] = 0
