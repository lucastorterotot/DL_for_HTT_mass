

default_attrs = ["PT", "Eta", "Phi"]
def store(dic, name, obj, attrs = default_attrs):
    if obj is None:
        for attr in attrs:
            dic["{}_{}".format(name, attr)] = 0
    else:
        for attr in attrs:
            dic["{}_{}".format(name, attr)] = getattr(obj, attr, 0)

            
default_gen_ptc_attrs = ["PID", "IsPU", "Charge", "Mass", "E", "PT", "Eta", "Phi", "D0", "DZ"]
def store_gen_ptc(dic, name, gen_ptc, attrs = default_gen_ptc_attrs):
    store(dic, name, gen_ptc, attrs = attrs)

    
default_jet_attrs = ["Mass", "PT", "Eta", "Phi", "Flavor", "BTag"]
def store_jet(dic, name, jet, attrs = default_jet_attrs):
    if jet is None:
        for attr in attrs:
            dic["{}_{}".format(name, attr)] = 0
    else:
        store(dic, name, jet, attrs = attrs)

default_real_tau_decays_attrs = list(set(default_attrs+["Charge"]+default_jet_attrs))
default_real_tau_decays_attrs.remove("Flavor")
default_real_tau_decays_attrs.remove("BTag")
def store_real_tau_decays(dic, name, decays, type=None, attrs = default_real_tau_decays_attrs):
    if type == "t":
        store_tauh(dic, name, decays, attrs=attrs)
    elif type == "m":
        store_muon(dic, name, decays, attrs=attrs)
    elif type =="e":
        store_electron(dic, name, decays, attrs=attrs)

default_tauh_attrs = default_jet_attrs
default_tauh_attrs.remove("Flavor")
default_tauh_attrs.remove("BTag")
def store_tauh(dic, name, jet,  attrs = default_tauh_attrs):
    store_jet(dic, name, jet, attrs = attrs)
    if dic["{}_{}".format(name, "Mass")] == None:
        dic["{}_{}".format(name, "Mass")] = 1776.86*10**(-3)

def store_muon(dic, name, muon, attrs = default_attrs+["Charge"]):
    store(dic, name, muon, attrs = attrs)
    dic["{}_{}".format(name, "Mass")] = 105.6583745*10**(-3)

def store_electron(dic, name, electron, attrs = default_attrs+["Charge"]):
    store(dic, name, electron, attrs = attrs)
    dic["{}_{}".format(name, "Mass")] = 0.5109989461*10**(-3)
