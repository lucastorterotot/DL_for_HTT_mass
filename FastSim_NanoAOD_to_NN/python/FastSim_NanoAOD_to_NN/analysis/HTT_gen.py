import FastSim_NanoAOD_to_NN.modules.store_vars as store_vars

def find_HTT(evt):
    ''' Find the Higgs boson in the generated particles,
    check that it decays in tau leptons,
    return the Higgs and the two taus.'''
    Higgs_IDs = [25, 35, 36]
    tau_ID = 15
    gen_taus_indexes = [k for k in range(int(evt.GetLeaf("nGenPart").GetValue(0))) if abs(evt.GetLeaf("GenPart_pdgId").GetValue(k)) == tau_ID]
    Higgs_decaying_into_taus_indexes = [int(evt.GetLeaf("GenPart_genPartIdxMother").GetValue(k)) for k in gen_taus_indexes]
    if len(gen_taus_indexes) == 2 and Higgs_decaying_into_taus_indexes[0] == Higgs_decaying_into_taus_indexes[1]:
        return Higgs_decaying_into_taus_indexes[0], gen_taus_indexes[0], gen_taus_indexes[1]
    return None, None, None
                        
def HTT_analysis(evt, verbose = 0, fast=True):
    output = {}
    Higgs, tau1, tau2 = find_HTT(evt)
    if not Higgs :
        return output

    store_vars.store_gen_ptc(evt, output, "Higgs", Higgs)
    store_vars.store_gen_ptc(evt, output, "tau1", tau1)
    store_vars.store_gen_ptc(evt, output, "tau2", tau2)

    if verbose > 2:
        print("\tHiggs energy is {} GeV.".format(Higgs.E))

    store_vars.store_gen_MET(evt, output)

    return output
