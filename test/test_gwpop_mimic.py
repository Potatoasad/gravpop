from gravpop import *


def test_gwpop_mimic():
	gwpop = GWPopLoader(posterior_file = "/Users/asadh/Documents/Data/posteriors.pkl",
						prior_file = "/Users/asadh/Documents/Data/production.prior",
    					vt_file = "/Users/asadh/Downloads/o1+o2+o3_bbhpop_real+semianalytic-LIGO-T2100377-v2.hdf5")
	gwpop.load_prior();
	gwpop.load_model();
	gwpop.load_vt();
	gwpop.load_posteriors();
	gwpop.create_likelihood();
