#==============================================================================
# This sample script reproduces the energy and zenith distributions for the
# nominal monte carlo and experimental data. As mentioned in the ReadMe,
# the file NuFSGenMC_nominal.dat contains an example flux,
# pre-calculated using the Honda+Gaisser pion and kaon flux model.
#
#==============================================================================

# First, we calculate the event weight via:
#     weight = flux * mcweight    ...(1)
# and plot the MC with the data

import sys
import numpy as np
import pylab
import functools
import tqdm


font = {'family':'serif',
        'serif':'Computer Modern Roman',
        'weight':200,
        'size':22}

pylab.rc('font', **font)

# Import the experimental data
class DataClass():
    def __init__(self,infile):
        self.name = infile.split("/")[-1]
        print('Loading '+str(infile.split("/")[-1])+' ...')
        try:
            expData = np.loadtxt(infile, comments="#")
            self.exp_energy = expData[:, 0]
            self.exp_cosZ = expData[:, 1]
        except:
            print('We could not find the file. Check paths.')

#Data = DataClass('../../data/observed_events.dat')


# import the monte carlo
class MCClass():
    def __init__(self, infile=None):
        if infile is not None:
            self.load(infile)

    def load(self, infile):
        self.name = infile.split("/")[-1]
        print('Loading '+str(infile.split("/")[-1])+' ...')
        mcData = np.loadtxt(infile, delimiter=' ', comments="#")
        self.load_array(mcData)

    def load_array(self, mcData):
        self.mcData = mcData
        self.pid = mcData[:, 0]
        self.reco_energy = mcData[:, 1]
        self.reco_cosZ = mcData[:, 2]
        self.true_energy = mcData[:, 3]
        self.true_cosZ = mcData[:, 4]
        self.mc_weight = mcData[:, 5]
        self.pion_flux = mcData[:, 6]
        self.kaon_flux = mcData[:, 7]
        self.total_flux = self.pion_flux + self.kaon_flux
        self.weights = self.total_flux * self.mc_weight
        self.total_count_nominal = self.weights


MonteCarlo = MCClass('NuFSGenMC_nominal.dat')

def create_fast_mc(mc, true_czbins, true_ebins, reco_czbins, reco_ebins, avg, add, match):
    variables = [mc.true_cosZ, mc.true_energy, mc.reco_cosZ, mc.reco_energy]
    binnings = [true_czbins, true_ebins, reco_czbins, reco_ebins]
    mappings = [np.digitize(v, bins=bins) for v, bins in zip(variables, binnings)]
    mappings = [m - np.amin(m) for m in mappings]
    mappings.extend([np.unique(mc.mcData[:, i], return_inverse=True)[1] for i in np.ix_(match)])
    masks = [[m == i for i in np.unique(m)] for m in mappings]
    grid_shape = tuple([np.amax(m) for m in mappings])
    new_mcData = []
    last_idxs = [None for i in range(len(grid_shape))]
    is_good = [False for i in range(len(grid_shape))]
    cached_masks = [None for i in range(len(grid_shape))]

    def get_mask(idx):
        good_idx = -1
        mask = None
        for i in range(len(grid_shape)-1, -1, -1):
            if last_idxs == idx[:i+1]:
                mask = cached_masks[i]
                good_idx = i
                if not is_good[i]:
                    return mask, False
                break
        if good_idx == -1:
            mask = masks[i][idx[0]]
            last_idxs[0] = idx[:1]
            cached_masks[0] = mask
            is_good[0] = np.count_nonzero(mask)
            if not is_good[0]:
                return mask, False
            good_idx = 0
        for j in range(i+1, len(grid_shape)):
            mask = np.logical_and(mask, masks[j][idx[j]])
            last_idxs[j] = idx[:j+1]
            cached_masks[j] = mask
            is_good[j] = np.count_nonzero(mask)
            if not is_good[j]:
                return mask, False
        return mask, True

    for i, idx in tqdm.tqdm(enumerate(np.ndindex(grid_shape)), total=np.prod(grid_shape)):
        mask = functools.reduce(np.logical_and, [m[i] for m, i in zip(masks, idx)])
        #mask, good = get_mask(idx)
        #if not good:
        #    continue
        if np.count_nonzero(mask) == 0:
            continue
        bin_mc = mc.mcData[mask]
        bin_weights = mc.weights[mask]
        new_event = np.zeros(mc.mcData.shape[1])
        new_event[add] = np.sum(bin_mc[:, add], axis=0)
        new_event[avg] = np.sum(bin_mc[:, avg] * bin_weights[:, None], axis=0) / np.sum(bin_weights)
        new_mcData.append(new_event)
    new_mc = MCClass()
    new_mcData = np.array(new_mcData)
    new_mc.load_array(new_mcData)
    return new_mc

def create_fast_mc(mc, true_czbins, true_ebins, reco_czbins, reco_ebins, avg, add, match):
    variables = [mc.true_cosZ, mc.true_energy, mc.reco_cosZ, mc.reco_energy]
    binnings = [true_czbins, true_ebins, reco_czbins, reco_ebins]
    mappings = [np.digitize(v, bins=bins) for v, bins in zip(variables, binnings)]
    mappings = [m - np.amin(m) for m in mappings]
    mappings.extend([np.unique(mc.mcData[:, i], return_inverse=True)[1] for i in np.ix_(match)])
    mappings = np.array(mappings)

    print(np.shape(mappings))

    u_mappings, mapping_inverse = np.unique(mappings, axis=1, return_inverse=True)
    print(np.shape(u_mappings))
    print(np.shape(mapping_inverse))
    n_new_events = np.shape(u_mappings)[1]
    new_mcData = np.zeros((n_new_events,) + mc.mcData.shape[1:], dtype=mc.mcData.dtype)
    new_weight_sum = np.zeros(n_new_events)

    for i in tqdm.tqdm(range(len(mc.mcData))):
        new_weight_sum[mapping_inverse[i]] += mc.weights[i]
        new_mcData[mapping_inverse[i],avg] += mc.mcData[i,avg] * mc.weights[i]
        new_mcData[mapping_inverse[i],add] += mc.mcData[i,add]
        new_mcData[mapping_inverse[i],match] = mc.mcData[i,match]
    new_mcData[:, avg] /= new_weight_sum[:, None]
    new_mc = MCClass()
    new_mcData = np.array(new_mcData)
    new_mc.load_array(new_mcData)
    return new_mc

avg = np.array([False, True, True, True, True, False, True, True])
add = np.array([False, False, False, False, False, True, False, False])
match = np.array([True, False, False, False, False, False, False, False])

num_cosz_bins = 20
cosz_low = -1.
cosz_high = 1.
num_e_bins = 80
log_e_low = 2
log_e_high = 6

cz_bins = np.linspace(cosz_low, cosz_high, num_cosz_bins + 1)
e_bins = np.logspace(log_e_low, log_e_high, num_e_bins + 1)

num_cosz_bins = 40
cosz_low = -1.
cosz_high = 1.
num_e_bins = 160
log_e_low = 0
log_e_high = 8

true_cz_bins = np.linspace(cosz_low, cosz_high, num_cosz_bins + 1)
true_e_bins = np.logspace(log_e_low, log_e_high, num_e_bins + 1)

new_mc = create_fast_mc(MonteCarlo, true_cz_bins, true_e_bins, cz_bins, e_bins, avg, add, match)

np.savetxt("fastmc.txt", new_mc.mcData)
