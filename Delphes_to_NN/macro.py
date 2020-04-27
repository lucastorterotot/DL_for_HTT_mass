#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from optparse import OptionParser
usage = "usage: %prog [options] <input root file> <output file name>"
parser = OptionParser(usage=usage)
parser.add_option("-v", "--verbose", dest = "verbose",
                                    default=0)
parser.add_option("-N", "--Nmax", dest = "Nmax",
                                    default=None)

(options,args) = parser.parse_args()

options.verbose = int(options.verbose)

if len(args) == 2:
    input_file_str = args[0]
    output_file_str = args[1]
else:
    raise RuntimeError("Please give one input file and one output file")

import ROOT
import libPyROOT as _root
ROOT.gSystem.Load("libDelphes")

input_file = ROOT.TFile(input_file_str, "READ")
tree = input_file.Get("Delphes")

if options.Nmax is None:
    Nmax = tree.GetEntries()
else:
    Nmax = min([int(options.Nmax), tree.GetEntries()])
                        
def match(ptc1, ptc2, dR = .5):
    ''' Check that two particles instance can be linked to the same physic object.'''
    if ptc1.PID != ptc2.PID:
        return False
    # if abs((ptc1.PT - ptc2.PT)/ptc1.PT) > .1:
    #     return False
    if DR2(ptc1, ptc2)**.5 > dR:
        return False
    return True

Nevt = 0

channel_stats_gen = {
    "tt":0,
    "mt":0,
    "et":0,
    "mm":0,
    "ee":0,
    "em":0,
}

channel_stats_reco = {}
channel_identification = {}

for c1 in channel_stats_gen.keys():
    channel_stats_reco[c1] = 0
    for c2 in channel_stats_gen.keys():
        channel_identification["{} as {}".format(c1, c2)] = 0

from Delphes_to_NN.analysis.HTT_gen import HTT_analysis as HTT_analysis_gen
from Delphes_to_NN.analysis.HTT_reco import HTT_analysis as HTT_analysis_reco

for evt in tree:
    Nevt += 1
    if options.verbose > 0:
        print("\nEvent {}:".format(Nevt))

    gen_analysis = HTT_analysis_gen(evt, verbose = options.verbose)
    channel_stats_gen[gen_analysis["channel"]] += 1

    reco_analysis = HTT_analysis_reco(evt, verbose = options.verbose)
    if reco_analysis != {}:
        channel_stats_reco[reco_analysis["channel"]] += 1
        channel_identification["{} as {}".format(
            gen_analysis["channel"],
            reco_analysis["channel"]
            )] += 1

    if options.verbose > 0:
        print("")
    if Nevt >= Nmax:
        break

print("Processed on {Nevt} events.".format(Nevt=Nevt))
for channel in channel_stats_gen:
    print("\t{} proportion: {} +/- {} pct".format(channel, 100*channel_stats_gen[channel]/Nevt, 100*channel_stats_gen[channel]**.5/Nevt))
    print("\t\t{} found = {} pct eff".format(channel_stats_reco[channel], 100*channel_stats_reco[channel]/channel_stats_gen[channel]))

for c1 in channel_stats_gen.keys():
    for c2 in channel_stats_gen.keys():
        channel_identification["{} as {}".format(c1, c2)] *= 1./(channel_stats_gen[c1])
print channel_identification
