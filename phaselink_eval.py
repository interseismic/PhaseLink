#! /home/zross/bin/python

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
import pylab as plt
from obspy import UTCDateTime
from geopy.distance import geodesic

# ! Make sure model has sigmoid(out) !

class StackedGRU(torch.nn.Module):
    def __init__(self):
        super(StackedGRU, self).__init__()
        self.hidden_size = 128
        self.fc1 = torch.nn.Linear(5, 32)
        self.fc2 = torch.nn.Linear(32, 32)
        self.fc3 = torch.nn.Linear(32, 32)
        self.fc4 = torch.nn.Linear(32, 32)
        self.fc5 = torch.nn.Linear(32, 32)
        self.fc6 = torch.nn.Linear(2*self.hidden_size, 1)
        self.gru1 = torch.nn.GRU(32, self.hidden_size, \
            batch_first=True, bidirectional=True)
        self.gru2 = torch.nn.GRU(self.hidden_size*2, self.hidden_size, \
            batch_first=True, bidirectional=True)

    def forward(self, inp):
        out = self.fc1(inp)
        out = torch.nn.functional.relu(out)
        out = self.fc2(out)
        out = torch.nn.functional.relu(out)
        out = self.fc3(out)
        out = torch.nn.functional.relu(out)
        out = self.fc4(out)
        out = torch.nn.functional.relu(out)
        out = self.fc5(out)
        out = torch.nn.functional.relu(out)
        out = self.gru1(out)
        h_t = out[0]
        out = self.gru2(h_t)
        h_t = out[0]
        out = self.fc6(h_t)
        out = torch.sigmoid(out)
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


def link_phases(Y, n_min_nucl, n_min_merge):
    clusters = []

    for i in range(Y.shape[0]):

        idx = np.where(Y[i,:] == 1)[0]
        if idx.size < n_min_nucl:
            continue
        idx += i
        idx = idx[np.where(idx < Y.shape[0])[0]]
        idx_set = set(idx)

        if len(clusters) == 0:
            clusters.append(idx_set)
            continue

        n_common = np.zeros(len(clusters))
        for j, cluster in enumerate(clusters):
            n_common[j] = len(cluster.intersection(idx_set))
        best = np.argmax(n_common)

        if n_common[best] < n_min_merge:
            clusters.append(idx_set)
        else:
            clusters[best].update(idx_set)

    return np.array(clusters)

def repick_event(cluster, X, params):

    idx = np.array(list(cluster))
    idx = idx[np.where(idx < X.shape[0])[0]]
    t_start = np.min(X[idx,2]) - params['t_repick']/2.0
    t_stop = np.max(X[idx,2]) + params['t_repick']/2.0

    return np.where((X[:,2] >= t_start) & (X[:,2] < t_stop))[0].astype(np.int32)


from numba import jit
@jit(nopython=True)
def back_project(cluster, X, indices, tt_p, tt_s,
                 max_pick_dist, phases, min_sep):
    best_cluster0 = np.zeros(cluster.size, dtype=np.int32)
    best_cluster = np.zeros(cluster.size, dtype=np.int32)
    arrival_times = np.zeros(cluster.size, dtype=np.float64)
    weights = np.zeros(cluster.size, dtype=np.int32)
    n_best = 0

    for i in range(tt_p.shape[1]):
        for j in range(tt_p.shape[2]):
            for k in range(tt_p.shape[3]):

                for l in range(cluster.size):
                    if phases[l]:
                        arrival_times[l] = tt_s[indices[l],i,j,k]
                        if arrival_times[l] <= max_pick_dist / 3.5:
                            weights[l] = 1
                        else:
                            weights[l] = 0
                    else:
                        arrival_times[l] = tt_p[indices[l],i,j,k]
                        if arrival_times[l] <= max_pick_dist / 6.0:
                            weights[l] = 1
                        else:
                            weights[l] = 0

                tt_diff = X[cluster,2] - arrival_times

                n_best2 = 0
                for l in range(len(cluster)):
                    start = tt_diff[l]
                    stop = start + min_sep
                    idx = np.where(
                        np.logical_and(tt_diff >= start, tt_diff < stop)
                    )[0]
                    if np.sum(weights[idx]) > n_best2:
                        best_cluster0[:] = 0
                        best_cluster0[idx] = 1
                        best_cluster0 *= weights
                        n_best2 = np.sum(best_cluster0)

                if n_best2 > n_best:
                    n_best = n_best2
                    best_cluster[:] = best_cluster0

    return cluster[np.where(best_cluster==1)[0]]

class tt_interp:
    def __init__(self, ttfile, datum):
        with open(ttfile, 'r') as f:
            count = 0
            for line in f:
                if count == 0:
                    count += 1
                    continue
                elif count == 1:
                    n_dist, n_depth = line.split()
                    n_dist = int(n_dist)
                    n_depth = int(n_depth)
                    dists = np.zeros(n_dist)
                    tt = np.zeros((n_depth, n_dist))
                    count += 1
                    continue
                elif count == 2:
                    depths = line.split()
                    depths = np.array([float(x) for x in depths])
                    count += 1
                    continue
                else:
                    temp = line.split()
                    temp = np.array([float(x) for x in temp])
                    dists[count-3] = temp[0]
                    tt[:, count-3] = temp[1:]
                    count += 1
        self.tt = tt
        self.dists = dists
        self.depths = depths
        self.datum = datum

        from scipy.interpolate import RectBivariateSpline
        self.interp_ = RectBivariateSpline(self.depths, self.dists, self.tt)

    def interp(self, dist, depth):
        return self.interp_.ev(depth + self.datum, dist)



def build_tt_grid(params):

    NX = params['n_x_nodes']
    NY = params['n_y_nodes']
    NZ = params['n_z_nodes']
    x = np.linspace(params['lon_min'], params['lon_max'], NX)
    y = np.linspace(params['lat_min'], params['lat_max'], NY)
    z = np.linspace(params['z_min'], params['z_max'], NZ)

    # Pwaves
    pTT = tt_interp(params['tt_table']['P'], params['datum'])
    print('Read pTT')
    print('(dep,dist) = (0,0), (10,0), (0,10), (10,10):')
    print('             {:.3f}   {:.3f}   {:.3f}   {:.3f}'.format(
        pTT.interp(0,0).item(), pTT.interp(10,0).item(),pTT.interp(0,10).item(),
        pTT.interp(10,10).item()))

    #Swaves
    sTT = tt_interp(params['tt_table']['S'], params['datum'])
    print('Read sTT')
    print('(dep,dist) = (0,0), (10,0), (0,10), (10,10):')
    print('             {:.3f}   {:.3f}   {:.3f}   {:.3f}'.format(
        sTT.interp(0,0).item(), sTT.interp(10,0).item(),sTT.interp(0,10).item(),
        sTT.interp(10,10).item()))

    n_stations = 0
    with open(params['station_file']) as f:
        for line in f:
            n_stations += 1

    tt_p = np.zeros((n_stations, NY, NX, NZ), dtype=np.float32)
    tt_s = np.zeros((n_stations, NY, NX, NZ), dtype=np.float32)
    station_index_map = {}

    n_stations = 0
    with open(params['station_file']) as f:
        for line in f:
            try:
                net, sta, lat, lon, elev = line.split()
            except:
                net, sta, lat, lon = line.split()
            station_index_map[(net, sta)] = n_stations
            print(n_stations)
            for i in range(NX):
                for j in range(NY):
                    dist = geodesic((y[i], x[j]), (lat, lon)).km
                    for k in range(NZ):
                        tt_p[n_stations, j, i, k] = pTT.interp(dist, z[k])
                        tt_s[n_stations, j, i, k] = sTT.interp(dist, z[k])
            n_stations += 1

    return tt_p, tt_s, station_index_map


#def build_tt_arrays(inputs, pTT, sTT, x, y, z):
#    net, sta, lat, lon = inputs
#    tt_p = np.zeros((y.size, x.size, z.size))
#    tt_s = np.zeros((y.size, x.size, z.size))
#    print(n_stations)
#    for i in range(NX):
#        for j in range(NY):
#            dist = geodesic((y[i], x[j]), (lat, lon)).km
#            for k in range(NZ):
#                tt_p[j, i, k] = pTT.interp(dist, z[k])
#                tt_s[j, i, k] = sTT.interp(dist, z[k])
#    return 


def build_idx_maps(labels, new_clust, station_index_map):
    phase_idx = {'P': 0, 'S': 1}
    indices = []
    phases = []
    for idx in new_clust:
        net, sta, phase, _ = labels[idx].split()
        indices.append(station_index_map[(net, sta)])
        phases.append(phase_idx[phase])
    phases = np.array(phases, dtype=np.int32)
    indices = np.array(indices, dtype=np.int32)
    return phases, indices

def run_phaselink(X, labels, trig_pr, params, ofile, tt_p, tt_s,
                  station_index_map):
    import time

    # Permute pick matrix for all lags 
    print("Permuting sequence for all lags...")
    X_perm = permute_seq(X, params['t_win'], params['n_max_picks'])
    X_perm = torch.from_numpy(X_perm).float().to(device)
    print("Finished permuting sequence")

    # Predict association labels for all windows
    Y_pred = torch.zeros((X_perm.size(0), X_perm.size(1), 1)).float()
    Y_pred = Y_pred.to(device)
    print("Predicting labels for all phases")
    for i in range(0, Y_pred.shape[0], params['batch_size']):
        i_start = i
        i_stop = i + params['batch_size']
        if i_stop > Y_pred.shape[0]:
            i_stop = Y_pred.shape[0]
        X_test = X_perm[i_start:i_stop]
        #X_test = X_test.cuda(device)
        with torch.no_grad():
            Y_pred[i_start:i_stop] = model(X_test)
    Y_pred = Y_pred.view(Y_pred.size(0), Y_pred.size(1))
    print("Finished label prediction")

    Y0 = torch.round(Y_pred).cpu().numpy()

    print("Linking phases")
    clusters = link_phases(Y0, params['n_min_nucl'], params['n_min_merge'])
    print("%d events detected initially" % len(clusters))

    # Remove events below threshold
    print("Removing duplicates")
    for i, cluster in enumerate(clusters):
        phases = {}
        for idx in cluster:
            if idx >= len(labels):
                continue
            net, sta, phase, time = labels[idx].split()
            if (net, sta, phase) not in phases:
                phases[(net, sta, phase)] = [(idx, trig_pr[idx])]
            else:
                phases[(net, sta, phase)].append((idx, trig_pr[idx]))
        for key in phases:
            if len(phases[key]) > 1:
                sorted(phases[key], key=lambda x: x[1])
                phases[key] = [phases[key][-1]]
        clusters[i] = [phases[key][0][0] for key in phases]
    clusters = [x for x in clusters if len(x) >= params['n_min_det']]
    print("%d events detected after duplicate removal" % len(clusters))

    if params['back_project']:
        # Repick and back-project to clean up
        for i, cluster in enumerate(clusters):
            new_clust = repick_event(cluster, X, params)
            phases, indices = build_idx_maps(labels, new_clust, station_index_map)
            clusters[i] = back_project(
                new_clust, X, indices, tt_p, tt_s,
                params['max_pick_dist'], phases, params['min_sep']
            )
            print("Backproject: {} -> {} -> {} picks".format(
                len(cluster), len(new_clust), len(clusters[i])))

    # Remove events below threshold
    clusters = [x for x in clusters if len(x) >= params['n_min_det']]

    # Plot results
    print("{} events left after applying n_min_det threshold".format(
        len(clusters))
    )

    # Write out solutions
    for i, cluster in enumerate(clusters):
        idx = np.array(list(cluster))

        event = Event()
        for j in idx:
            net, sta, phase, time = labels[j].split()
            proba = trig_pr[j]
            #pick_error = proba_error_map(proba)
            pick_error = 0.10
            time = UTCDateTime(time)
            arrival = Arrival(net, sta, time, phase)
            event.arrivals.append(arrival)

            ofile.write(
                "%-6s %-4s %-4s %-1s %-6s %-1s "
                "%04d%02d%02d %02d%02d "
                "%7.4f %-3s %9.2e %9.2e %9.2e %9.2e\n" % \
                (
                    sta,
                    net,
                    "?",
                    "?",
                    phase,
                    "?",
                    time.year,
                    time.month,
                    time.day,
                    time.hour,
                    time.minute,
                    time.second+time.microsecond/1.e6,
                    "GAU",
                    pick_error,
                    -1,
                    -1,
                    -1,
                )
            )
        ofile.write("\n")

    return len(clusters)


def detect_events(X, Y, model, params):

    ofile = open(params['outfile'], 'w')

    X[:,2] -= X[0,2]
    trig_pr_full = X[:,5]
    X = X[:,:5]

    n_cumul_dets = 0

    if params['back_project']:
        tt_p, tt_s, station_index_map = build_tt_grid(params)
    else:
        tt_p, tt_s, station_index_map = None, None, None

    for t_start in np.arange(0, X[-1,2], 86400.):

        t_stop = t_start + 86400.

        idx = np.where((X[:,2] >= t_start) & (X[:,2] < t_stop))[0]

        print("Day {:03d}: {:d} gpd picks, {:d} "
              "cumulative events detected".format(
              int(t_start/86400.)+1, idx.size, n_cumul_dets))

        if idx.size == 0:
            continue

        labels = [Y[x] for x in idx]
        trig_pr = trig_pr_full[idx]

        n_cumul_dets += run_phaselink(
            X[idx], labels, trig_pr, params, ofile, tt_p, tt_s,
            station_index_map
        )

    print("{} detections total".format(n_cumul_dets))

    ofile.close()

def read_gpd_output(params):
    from obspy import UTCDateTime
    print("Reading GPD file")
    stations = pickle.load(open(params['station_map_file'], 'rb'))
    sncl_map = pickle.load(open(params['sncl_map_file'], 'rb'))
    X = []
    labels = []
    phase_idx = {'P': 0, 'S': 1}
    missing_sta_list = set()
    with open(params['gpd_file'], 'r') as f:
        for line in f:
            net, sta, phase, time, pr, dur = line.split()
            if float(pr) < params['pr_min']:
                continue
            if float(dur) < params['trig_dur_min']:
                continue
            try:
                sta_X, sta_Y = stations[(net, sta, phase)]
            except:
                if (net, sta) not in missing_sta_list:
                    print("%s %s missing from station list" % (net, sta))
                    missing_sta_list.add((net, sta))
                continue
            otime = UTCDateTime("%s" % time) - UTCDateTime(0)
            X.append((sta_X, sta_Y, otime, phase_idx[phase], 1, float(pr)))
            labels.append("%s %s %s %s" % (net, sta, phase, time))
            if len(X) % 200000 == 0:
                print(len(X))
            if params['stop_read_early']:
                if len(X) == 200000:
                    print("Stopped reading early")
                    break
    X = np.array(X)
    idx = np.argsort(X[:,2])
    X = X[idx,:]
    labels = [labels[i] for i in idx]
    print("Finished reading GPD file, %d triggers found" % len(labels))

    return X, labels


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("phaselink_eval config_json")
        sys.exit()

    with open(sys.argv[1], "r") as f:
        params = json.load(f)

    device = torch.device(params['device'])

    model = StackedGRU().cuda(device)

    checkpoint = torch.load(params['model_file'], map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    X, labels = read_gpd_output(params)
    detect_events(X, labels, model, params)
