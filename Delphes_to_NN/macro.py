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


def DR2(ptc1, ptc2):
    return (ptc1.Eta - ptc2.Eta)**2 + (ptc1.Phi - ptc2.Phi)**2
                        
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
channel_stats_reco = {
    "tt":0,
    "mt":0,
    "et":0,
    "mm":0,
    "ee":0,
    "em":0,
}

from analysis.HTT_gen import HTT_analysis as HTT_analysis_gen

for evt in tree:
    Nevt += 1
    if options.verbose > 0:
        print("\nEvent {}:".format(Nevt))

    gen_analysis = HTT_analysis_gen(evt, verbose = options.verbose)
    channel_stats_gen[gen_analysis["channel"]] += 1

    if options.verbose > 0:
        print("")
    if Nevt >= Nmax:
        break

print("Processed on {Nevt} events.".format(Nevt=Nevt))
for channel in channel_stats_gen:
    print("\t{} proportion: {} +/- {} pct".format(channel, 100*channel_stats_gen[channel]/Nevt, 100*channel_stats_gen[channel]**.5/Nevt))
