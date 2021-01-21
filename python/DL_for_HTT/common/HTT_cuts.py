# HTT analysis cuts

allowed_Tau_decayMode = [0, 1, 2, 10, 11]

cut_tauh_tt_pt = 40
cut_tauh_tt_eta = 2.1
cut_tauh_tt_dz = 0.2

cut_tauh_mt_pt = 30
cut_tauh_mt_eta = 2.3
cut_tauh_mt_dz = cut_tauh_tt_dz

cut_tauh_et_pt = cut_tauh_mt_pt
cut_tauh_et_eta = cut_tauh_mt_eta
cut_tauh_et_dz = cut_tauh_mt_dz

cut_muon_mt_pt = 21
cut_muon_mt_eta = 2.1
cut_muon_mt_d0 = 0.045
cut_muon_mt_dz = cut_tauh_tt_dz

cut_muon_em_pt = 13
cut_muon_em_eta = 2.4
cut_muon_em_d0 = cut_muon_mt_d0
cut_muon_em_dz = cut_tauh_tt_dz

cut_muon_mm_pt = 10
cut_muon_mm_eta = 2.4
cut_muon_mm_d0 = cut_muon_mt_d0
cut_muon_mm_dz = cut_tauh_tt_dz

cut_iso_lepton = 0.3

cut_muon_mt_dilepton_veto_pt = 15
cut_muon_mt_dilepton_veto_eta = max([cut_muon_mt_eta, cut_muon_em_eta, cut_muon_mm_eta])
cut_muon_mt_dilepton_veto_d0 = cut_muon_mt_d0
cut_muon_mt_dilepton_veto_dz = cut_tauh_tt_dz
cut_muon_mt_dilepton_veto_iso = cut_iso_lepton

cut_muon_third_lepton_veto_pt = 10
cut_muon_third_lepton_veto_eta = max([cut_muon_mt_eta, cut_muon_em_eta, cut_muon_mm_eta])
cut_muon_third_lepton_veto_d0 = cut_muon_mt_d0
cut_muon_third_lepton_veto_dz = cut_tauh_tt_dz
cut_muon_third_lepton_veto_iso = cut_iso_lepton

cut_ele_et_pt = 25
cut_ele_et_eta = 2.1
cut_ele_et_d0 = cut_muon_mt_d0
cut_ele_et_dz = cut_tauh_tt_dz
cut_ele_et_iso = 0.15

cut_ele_em_pt = 13
cut_ele_em_eta = 2.5
cut_ele_em_d0 = cut_muon_mt_d0
cut_ele_em_dz = cut_tauh_tt_dz
cut_ele_em_iso = cut_ele_et_iso

cut_ele_ee_pt = 20
cut_ele_ee_eta = 2.5
cut_ele_ee_d0 = cut_muon_mt_d0
cut_ele_ee_dz = cut_tauh_tt_dz
cut_ele_ee_iso = 0.1

cut_ele_et_dilepton_veto_pt = 15
cut_ele_et_dilepton_veto_eta = max([cut_ele_et_eta, cut_ele_em_eta, cut_ele_ee_eta])
cut_ele_et_dilepton_veto_d0 = cut_muon_mt_d0
cut_ele_et_dilepton_veto_dz = cut_tauh_tt_dz
cut_ele_et_dilepton_veto_iso = cut_iso_lepton

cut_ele_third_lepton_veto_pt = 10
cut_ele_third_lepton_veto_eta = max([cut_ele_et_eta, cut_ele_em_eta, cut_ele_ee_eta])
cut_ele_third_lepton_veto_d0 = cut_muon_mt_d0
cut_ele_third_lepton_veto_dz = cut_tauh_tt_dz
cut_ele_third_lepton_veto_iso = cut_iso_lepton
