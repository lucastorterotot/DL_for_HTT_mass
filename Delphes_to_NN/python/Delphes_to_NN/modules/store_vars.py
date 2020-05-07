

default_attrs = ["PT", "Eta", "Phi"]
def store(dic, name, obj, attrs = default_attrs):
    if obj is None:
        for attr in attrs:
            dic["{}_{}".format(name, attr)] = None
    else:
        for attr in attrs:
            dic["{}_{}".format(name, attr)] = getattr(obj, attr, None)

            
default_gen_ptc_attrs = ["PID", "IsPU", "Charge", "Mass", "E", "PT", "Eta", "Phi", "D0", "DZ"]
def store_gen_ptc(dic, name, gen_ptc, attrs = default_gen_ptc_attrs):
    store(dic, name, gen_ptc, attrs = attrs)

    
default_jet_attrs = ["Mass", "PT", "Eta", "Phi", "Flavor", "BTag"]
def store_jet(dic, name, jet, attrs = default_jet_attrs):
    store(dic, name, jet, attrs = attrs)

def store_real_tau_decays(dic, name, decays, type=None):
    if type == "t":
        store_tauh(dic, name, decays)
    elif type == "m":
        store_muon(dic, name, decays)
    elif type =="e":
        store_electron(dic, name, decays)

default_tauh_attrs = default_jet_attrs
default_tauh_attrs.remove("Flavor")
default_tauh_attrs.remove("BTag")
def store_tauh(dic, name, jet):
    store_jet(dic, name, jet, attrs = default_tauh_attrs)

def store_muon(dic, name, muon, attrs = default_attrs+["Charge"]):
    store(dic, name, muon, attrs = attrs)

def store_electron(dic, name, electron, attrs = default_attrs+["Charge"]):
    store(dic, name, electron, attrs = attrs)
