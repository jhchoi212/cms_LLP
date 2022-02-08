import ROOT, sys
import numpy as np

inFileName = sys.argv[1]
print("Reading from " + str(inFileName))

inFile = ROOT.TFile.Open(inFileName,"READ")

tree = inFile.Get("ntuple0/objects")
ver = inFile.Get("ntuple0/objects/vz")
tree.show()
