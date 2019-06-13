#! /bin/env python

# PhaseLink: Earthquake phase association with deep learning
# Author: Zachary E. Ross
# Seismological Laboratory
# California Institute of Technology

import sys
import numpy as np
import pickle
import json
import torch
import torch.utils.data

class StackedGRU(torch.nn.Module):
    def __init__(self):
        super(StackedGRU, self).__init__()
        self.hidden_dim = 64
        self.gru1 = torch.nn.GRU(5, self.hidden_dim, \
            batch_first=True, bidirectional=True)
        self.gru2 = torch.nn.GRU(self.hidden_dim*2, self.hidden_dim, \
            batch_first=True, bidirectional=True)
        self.fc1 = torch.nn.Linear(2*self.hidden_dim, self.hidden_dim)
        self.fc2 = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = torch.nn.Linear(self.hidden_dim, 1)
        self.sigmoid = torch.nn.Sigmoid()
        #self.dropout = torch.nn.Dropout(p=0.25)

    def forward(self, x):
        out = self.gru1(x)
        h_t = out[0]
        out = self.gru2(h_t)
        h_t = out[0]
        out = self.fc1(h_t)
        out = torch.nn.functional.relu(out)
        out = self.fc2(out)
        out = torch.nn.functional.relu(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out

class Arrival():
    def __init__(self, net, sta, time, phase):
        self.net = net
        self.sta = sta
        self.time = time
        self.phase = phase

class Event():
    def __init__(self, arrivals = None):
        if arrivals is not None:
            self.arrivals = arrivals
        else:
            self.arrivals = []

def permute_seq(X, t_win, n_max_picks):
    X0 = np.zeros((X.shape[0], n_max_picks, X.shape[1]))
    for i in range(X.shape[0]):
        i_start = i
        i_end = i + n_max_picks
        if i_end > X.shape[0]:
            i_end = X.shape[0]

        # Map picks for slice into new array
        X0[i_start,:(i_end-i_start),:] = X[i_start:i_end,:]

        # Set initial pick to t=0
        idx = np.where(X0[i,:,2] > 0)[0]
        X0[i,idx,2] -= X0[i,0,2]

        # Remove all times with t > t_win
        idx = np.where(X0[i,:,2] > t_win)[0]
        X0[i,idx,:] = 0

        # Normalize time values
        X0[i,:,2] /= t_win

    return X0

def link_phases(Y, sncl_map, n_min_nucl, n_min_merge):
    clusters = []
    sncls = []
    master_pool = set()
    for i in range(Y.shape[0]-1, -1, -1):
        idx = np.where(Y[i,:] == 1)[0]
        if idx.size < n_min_nucl:
            continue
        idx += i
        idx_set = set(idx)
        sncls_set = set([(x, sncl_map[x]) for x in idx if x < len(sncl_map)])
        if len(master_pool.intersection(idx_set)) == 0:
            clusters.append(idx_set)
            sncls.append(sncls_set)
            master_pool.update(idx_set)
        else:
            best_cluster = 0
            best_count = 0
            j_vals = []
            n_best = 0
            for cluster in clusters:
                count = len(cluster.intersection(idx_set))
                j_vals.append(count)
            best_cluster = np.argmax(j_vals)
            if j_vals[best_cluster] < n_min_merge:
                continue

            # BEGIN SECTION FOR PREVENTING DUPLICATES
            #others = sncls[best_cluster].symmetric_difference(sncls_set)
            #other_sncls = set([x[1] for x in others])
            #count_sncl = len(others) - len(other_sncls)
            #if count_sncl > 0:
            #    continue
            # END SECTION FOR PREVENTING DUPLICATES

            clusters[best_cluster].update(idx_set)
            sncls[best_cluster].update(sncls_set)
            master_pool.update(idx_set)
    return clusters

def detect_events(X, labels, model, params):
    X[:,2] -= X[0,2]
    sncl_map = X[:,5]
    X = X[:,:5]

    # Permute pick matrix for all lags and drop picks more than t_win outside
    print("Permuting sequence for all lags...")
    X_perm = torch.from_numpy(permute_seq(X, params['t_win'],
        params['n_max_picks'])).float()
    print("Finished permuting sequence")

    # Predict association labels for all windows
    Y_pred = torch.zeros((X_perm.size(0), X_perm.size(1), 1)).float()
    print("Predicting labels for all phases")
    for i in range(0, Y_pred.shape[0], params['batch_size']):
        i_start = i
        i_stop = i + params['batch_size']
        if i_stop > Y_pred.shape[0]:
            i_stop = Y_pred.shape[0]
        X_test = X_perm[i_start:i_stop]
        X_test = X_test.cuda(device)
        Y_pred[i_start:i_stop] = model(X_test).detach()
    Y_pred = Y_pred.view(Y_pred.size(0), Y_pred.size(1))
    print("Finished label prediction")

    Y0 = torch.round(Y_pred).numpy()

    print("Linking phases")
    clusters = link_phases(Y0, sncl_map, params['n_min_nucl'],
        params['n_min_merge'])

    # Plot results
    print("%d events detected" % len(clusters))

    ofile = open(params['outfile'], 'w')
    from obspy import UTCDateTime
    events = []
    for i, cluster in enumerate(clusters):
        idx = np.array(list(cluster))
        phases = set()

        event = Event()
        for j in idx:
            #ofile.write("%d %s\n" % (i+1, labels[j]))
            net, sta, phase, time = labels[j].split()
            time = UTCDateTime(time)
            arrival = Arrival(net, sta, time, phase)
            event.arrivals.append(arrival)

            if (net, sta, phase) in phases:
                continue
            phases.add((net, sta, phase))
            ofile.write("%-6s %-4s %-4s %-1s %-6s %-1s "
                        "%04d%02d%02d %02d%02d "
                        "%7.4f %-3s %9.2e %9.2e %9.2e %9.2e\n"
                % (sta, "?", "?", "?", phase, "?", time.year, time.month,
                   time.day, time.hour, time.minute,
                   time.second+time.microsecond/1.e6, "GAU", 0, 0, 0, 0))
        ofile.write("\n")
        events.append(event)
    ofile.close()

    ofile = open(params['outpkl'], 'wb')
    pickle.dump(events, ofile)
    ofile.close()

def read_gpd_output(params):
    from obspy import UTCDateTime
    print("Reading GPD file")
    stations = pickle.load(open(params['station_map_file'], 'rb'))
    sncl_map = pickle.load(open(params['sncl_map_file'], 'rb'))
    X = []
    outputs = []
    phase_idx = {'P': 0, 'S': 1}
    with open(params['gpd_file'], 'r') as f:
        for line in f:
            net, sta, phase, time, pr, dur = line.split()
            if float(pr) < params['pr_min']:
                continue
            if float(dur) < params['trig_dur_min']:
                continue
            sta_X, sta_Y = stations[(net, sta, phase)]
            otime = UTCDateTime("%s" % time) - UTCDateTime(0)
            sncl_idx = sncl_map[(net, sta, phase)]
            X.append((sta_X, sta_Y, otime, phase_idx[phase], 1, sncl_idx))
            outputs.append("%s %s %s %s" % (net, sta, phase, time))
    X = np.array(X)
    idx = np.argsort(X[:,2])
    X = X[idx,:]
    outputs = [outputs[i] for i in idx]
    print("Finished reading GPD file, %d triggers found" % len(outputs))

    return X, outputs


    def predict(self, data_loader):
        from torch.autograd import Variable
        import time

        for inputs, labels in val_loader:

            # Wrap tensors in Variables
            inputs, labels = Variable(
                inputs.cuda(device)), Variable(
                labels.cuda(device))

            # Forward pass only
            val_outputs = self.network(inputs)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("phaselink_eval config_json")
        sys.exit()

    with open(sys.argv[1], "r") as f:
        params = json.load(f)

    device = torch.device(params['device'])

    model = StackedGRU().cuda(device)

    checkpoint = torch.load(params['model_file'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    X, labels = read_gpd_output(params)
    detect_events(X, labels, model, params)
