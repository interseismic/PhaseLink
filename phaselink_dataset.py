#!/home/zross/bin/python
import numpy as np
import multiprocessing as mp
import pickle
import sys
import json
from obspy.geodetics.base import gps2dist_azimuth

bounds_scaler = 1
limit_max_distance = True
random_sta_locs = False

def output_thread(out_q, params):
    none_count = 0
    X = []
    Y = []
    while True:
        res = out_q.get()
        if res is None:
            none_count += 1
        else:
            X.append(res[0])
            Y.append(res[1])
            print(len(Y))
        if none_count == n_threads:
            break
    X = np.array(X)
    Y = np.array(Y)

    ones = np.sum(Y)
    zeros = np.size(Y) - ones
    total = ones + zeros
    print("Ones:", ones, 100*ones/total)
    print("Zeros:", zeros, 100*zeros/total)

    np.save(params["training_dset_X"], X)
    np.save(params["training_dset_Y"], Y)

    return

def generate_phases(in_q, out_q, x_min, x_max, y_min, y_max, \
                    sncl_idx, stla, stlo, phasemap, tt_p, tt_s):

    np.random.seed()

    # Random phase station time generator
    n_sta = sncl_idx.size

    while True:
        # Infinite loop until finished
        next_id = in_q.get()
        if next_id is None:
            out_q.put(None)
            #print("Job finished...exiting")
            break

        # Random phase station time generator
        n_sta = sncl_idx.size

        # Define random number of events in the window
        #n_eve = int(t_max/avg_t_sep)
        n_eve = int(t_max/params['avg_eve_sep'])

        # Generate random origin times
        origin_times = np.random.uniform(0, t_max, size=n_eve)
        origin_times[0] = 0

        # Define random number of fake picks in the window
        if random_sta_locs:
            phasemap = np.random.randint(0, 2, size=n_sta)
            stlo = np.random.uniform(x_min, x_max, size=n_sta)
            stla = np.random.uniform(y_min, y_max, size=n_sta)
        X = []
        Y = []
        R = []

        n_fake = params['n_fake']
        if n_fake > 0:

            if np.random.rand() < 0.5:
                idx = np.ones(n_fake, dtype=np.int) * \
                    np.random.choice(np.arange(stlo.size))
            else:
                idx = np.random.choice(np.arange(stlo.size),
                    size=n_fake,
                    replace=True)
            phase_labels = phasemap[idx]
            lons = stlo[idx]
            lats = stla[idx]
            tt = np.random.uniform(0, t_max, size=n_fake)
            X.append(np.column_stack((lons, lats, tt, phase_labels,
                np.ones(lons.shape[0]))))
            Y.append(np.zeros(lats.size))
            R.append(np.zeros(lats.size))

        Y0 = np.random.uniform(bounds_scaler*y_min,
            bounds_scaler*y_max)
        X0 = np.random.uniform(bounds_scaler*x_min,
            bounds_scaler*x_max)
        Z0 = np.random.uniform(0, params['max_event_depth'])

        for i in range(n_eve):

            # Calculate distances for all stations to source
            dists = np.sqrt((X0-stlo)**2 + (Y0-stla)**2)

            # Construct arrays for pick features, labels, and distances
            idx = np.where(phasemap == 0)[0]
            tt = np.zeros(dists.size)
            tt[idx] = tt_p.interp(dists[idx], np.tile(Z0, idx.size))
            phase_labels = phasemap
            idx = np.where(phasemap == 1)[0]
            tt[idx] = tt_s.interp(dists[idx], np.tile(Z0, idx.size))

            # Generate pick errors
            dt = params['max_pick_error']
            tt += np.random.uniform(-dt, dt, size=phasemap.size)

            #dt = np.random.normal(size=phasemap.size)*0.05*tt
            #idx = np.where(dt < 0.15)[0]
            #dt[idx] = 0.15
            #idx = np.where(dt > 1.50)[0]
            #dt[idx] = 1.50
            #tt += dt*tt

            idx = np.argsort(dists)
            tt = tt[idx]

            # Add back in origin time
            tt += origin_times[i]

            phase_labels = phase_labels[idx]
            dists = dists[idx]
            lons = stlo[idx]
            lats = stla[idx]

            max_dist = np.random.uniform(params['min_hypo_dist'],
                params['max_hypo_dist'])
            idx = np.where(dists <= max_dist)[0]
            tt = tt[idx]
            phase_labels = phase_labels[idx]
            dists = dists[idx]
            lons = lons[idx]
            lats = lats[idx]

            X.append(np.column_stack((lons, lats, tt, phase_labels,
                np.ones(lons.shape[0]))))
            Y.append(np.ones(lats.size)*i+1)
            R.append(dists)

        if len(X) == 0:
            continue

        X = np.concatenate(X)
        Y = np.concatenate(Y)
        R = np.concatenate(R)

        retain_pr = np.random.uniform(0, 1, size=Y.size)
        idx = np.where(retain_pr < 0.50)[0]
        X = X[idx,:]
        Y = Y[idx]
        R = R[idx]

        if len(X) == 0:
            continue

        idx = np.argsort(X[:,2])
        X = X[idx,:]
        Y = Y[idx]
        R = R[idx]

        X[:,2] -= X[0,2]

        # Check that the number of picks does not exceed the max allowed
        if X.shape[0] >= max_picks:
            Y = Y[:max_picks]
            X = X[:max_picks,:]
            R = R[:max_picks]
        else:
            Y.resize(max_picks)
            X_ = np.zeros((max_picks, 5))
            X_[X.shape[0]:,2] = 0.0
            X_[:X.shape[0], :] = X
            X = X_

        Y = Y.astype(np.int32)
        labels = np.unique(Y)

        X[:,0] = (X[:,0] - x_min) / (x_max - x_min)
        X[:,1] = (X[:,1] - y_min) / (y_max - y_min)
        X[:,2] /= t_max

        if Y[0] == 0:
            Y[:] = 0
            Y[0] = 1
        else:
            idx1 = np.where(Y == Y[0])[0]
            idx0 = np.where(Y != Y[0])[0]
            Y[idx0] = 0
            Y[idx1] = 1

        out_q.put((X, Y))

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


def get_network_centroid(params):
    stlo = []
    stla = []
    with open(params['station_file'], 'r') as f:
        for line in f:
            net, sta, lat, lon = line.split()
            stlo.append(float(lon))
            stla.append(float(lat))

    lat0 = (np.max(stla) + np.min(stla))*0.5
    lon0 = (np.max(stlo) + np.min(stlo))*0.5
    return lat0, lon0

def build_station_map(params):
    stations = {}
    sncl_map = {}
    count = 0
    with open(params['station_file'], 'r') as f:
        for line in f:
            net, sta, lat, lon = line.split()
            stla = float(lat)
            stlo = float(lon)
            X0 = gps2dist_azimuth(lat0, stlo, lat0, lon0)[0]/1000.
            if stlo < lon0:
                X0 *= -1
            Y0 = gps2dist_azimuth(stla, lon0, lat0, lon0)[0]/1000.
            if stla < lat0:
                Y0 *= -1
            if (net, sta, 'P') not in sncl_map:
                sncl_map[(net, sta, 'P')] = count
                stations[(net, sta, 'P')] = (X0, Y0)
                count += 1
            if (net, sta, 'S') not in sncl_map:
                sncl_map[(net, sta, 'S')] = count
                stations[(net, sta, 'S')] = (X0, Y0)
                count += 1
    stlo = np.array([stations[x][0] for x in sncl_map])
    stla = np.array([stations[x][1] for x in sncl_map])
    phasemap = np.array([phase_idx[x[2]] for x in sncl_map])
    sncl_idx = np.array([sncl_map[x] for x in sncl_map])

    return stlo, stla, phasemap, sncl_idx, stations, sncl_map


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("phaselink_dataset config_json")
        sys.exit()

    with open(sys.argv[1], "r") as f:
        params = json.load(f)

    max_picks = params['n_max_picks']
    t_max = params['t_win']
    n_threads = params['n_threads']

    print("Starting up...")
    phase_idx = {'P': 0, 'S': 1}

    lat0, lon0 = get_network_centroid(params)
    stlo, stla, phasemap, sncl_idx, stations, sncl_map = \
        build_station_map(params)

    x_min = np.min(stlo)
    x_max = np.max(stlo)
    y_min = np.min(stla)
    y_max = np.max(stla)

    for key in sncl_map:
        X0, Y0 = stations[key]
        X0 = (X0 - x_min) / (x_max - x_min)
        Y0 = (Y0 - y_min) / (y_max - y_min)
        stations[key] = (X0, Y0)

    # Save station maps for detect mode
    pickle.dump(stations, open(params['station_map_file'], 'wb'))
    pickle.dump(sncl_map, open(params['sncl_map_file'], 'wb'))

    in_q = mp.Queue(1000000)
    out_q = mp.Queue(1000000)

    # Pwaves
    pTT = tt_interp(params['tt_table']['P'], params['datum'])
    print('Read pTT')
    print('(dep,dist) = (0,0), (10,0), (0,10), (10,0):')
    print('             {:.3f}   {:.3f}   {:.3f}   {:.3f}'.format(
	pTT.interp(0,0), pTT.interp(10,0),pTT.interp(0,10),
	pTT.interp(10,10)))

    #Swaves
    sTT = tt_interp(params['tt_table']['S'], params['datum'])
    print('Read sTT')
    print('(dep,dist) = (0,0), (10,0), (0,10), (10,0):')
    print('             {:.3f}   {:.3f}   {:.3f}   {:.3f}'.format(
	sTT.interp(0,0), sTT.interp(10,0),sTT.interp(0,10),
	sTT.interp(10,10)))

    #pTT = tt_interp(params['trav_time_p'])
    #sTT = tt_interp(params['trav_time_s'])

    proc = mp.Process(target=output_thread, args=(out_q, params))
    proc.start()
    procs = []
    for i in range(n_threads):
        print("Starting thread %d" % i)
        p = mp.Process(target=generate_phases, \
            args=(in_q, out_q, x_min, x_max, y_min, y_max, \
                  sncl_idx, stla, stlo, phasemap, pTT, sTT))
        p.start()
        procs.append(p)

    for i in range(params['n_train_samp']):
        in_q.put(i)

    for i in range(n_threads):
        in_q.put(None)
    #for p in procs:
    #    p.join()
    #proc.join()
