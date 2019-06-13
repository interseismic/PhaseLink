#! /bin/env python
import numpy as np
import multiprocessing as mp
import pickle
import json

#np.random.seed(42)

#SOCAL
bounds_scaler = 2
n_chunk = 1000
max_picks = 200
n_samples = 1200000
t_max = 120.0
plot_seq = 0
n_threads = 16
n_rand_sta = 500

def output_thread(out_q, params):
    import h5py
    print("Building hdf5 dataset")
    f = h5py.File(params['hdf_file'], "w", libver='latest')
    f.create_dataset("X", (n_samples, max_picks, 5), \
                     dtype=np.float32)
    f.create_dataset("Y", (n_samples, max_picks), \
                     dtype=np.int16)
    print("Finished building hdf5 dataset")

    count = 0
    none_count = 0
    while True:
        res = out_q.get()
        if res is None:
            none_count += 1
        else:
            f['X'][count:count+n_chunk,:,:] = res[0]
            f['Y'][count:count+n_chunk,:] = res[1]
            print("chunk", count, "written")
            count += n_chunk
        if none_count == n_threads:
            break
    f.close()

def generate_phases(in_q, out_q, x_min, x_max, y_min, y_max, \
                    sncl_idx, stla, stlo, phasemap, tt_p, tt_s):
    np.random.seed()

    # Random phase station time generator
    #n_sta = sncl_idx.size
    n_sta = n_rand_sta

    while True:
        chunk = []
        # Infinite loop until finished
        next_id = in_q.get()
        if next_id is None:
            out_q.put(None)
            print("Job finished...exiting")
            break
        for k in range(n_chunk):

            # Define random number of events in the window
            n_eve = np.random.randint(0, 20)

            # Generate random origin times
            origin_times = np.random.uniform(0, 60, size=n_eve)
            #origin_times = np.cumsum(origin_times)

            # Define random number of fake picks in the window
            #n_fake = np.random.randint(0, max_picks//10)
            n_fake = 0

            phasemap = np.random.randint(0, 2, size=n_rand_sta)
            stlo = np.random.uniform(x_min, x_max, size=n_rand_sta)
            stla = np.random.uniform(y_min, y_max, size=n_rand_sta)

            feat_stlo = np.tile(stlo, n_eve)
            feat_stla = np.tile(stla, n_eve)
            label0 = np.ones(feat_stlo.size)
            feat01 = np.array([])
            feat02 = np.array([])
            dists = np.array([])
            for i in range(n_eve):
                Y0 = np.random.uniform(bounds_scaler*y_min,
                    bounds_scaler*y_max)
                X0 = np.random.uniform(bounds_scaler*x_min,
                    bounds_scaler*x_max)
                Z0 = np.random.uniform(0, 25)

                # Calculate distances for all stations to source
                dists0 = np.sqrt((X0-stlo)**2 + (Y0-stla)**2)

                # Generate pick errors
                dt = np.random.normal(0, 0.33, size=phasemap.size)

                # Construct arrays for pick features, labels, and distances
                idx = np.where(phasemap == 0)[0]
                tt = np.zeros(dists0.size)
                tt[idx] = tt_p.interp(dists0[idx], np.tile(Z0, idx.size))
                feat02_0 = phasemap
                idx = np.where(phasemap == 1)[0]
                tt[idx] = tt_s.interp(dists0[idx], np.tile(Z0, idx.size))
                tt += dt

                feat01 = np.concatenate((feat01, tt)) if feat01.size else tt
                feat02 = np.concatenate((feat02, feat02_0)) if feat02.size else feat02_0
                dists = np.concatenate((dists, dists0)) if dists.size else dists0

            if n_eve > 0:
                feat01 += np.repeat(origin_times, n_sta)
                label0 += np.repeat(np.arange(n_eve), n_sta)
                max_dist = np.repeat(np.random.uniform(40, 150, size=n_eve), n_sta)

                # Remove picks that are more than max_dist away
                idx = np.where(dists <= max_dist)[0]
                feat_stlo = feat_stlo[idx]
                feat_stla = feat_stla[idx]
                feat01 = feat01[idx]
                feat02 = feat02[idx]
                label0 = label0[idx]

                # Randomly flip the phase type and label to 0
                if True:
                    retain_pr = np.random.uniform(0, 1, size=label0.size)
                    idx = np.where(retain_pr < 0.10)[0]
                    idx2 = np.where(feat02[idx] == 0)[0]
                    idx3 = np.where(feat02[idx] == 1)[0]
                    feat02[idx[idx2]] = 1
                    feat02[idx[idx3]] = 0
                    label0[idx] = 0

                retain_pr = np.random.uniform(0, 1, size=label0.size)
                idx = np.where(retain_pr >= 0.10)[0]
                feat_stlo = feat_stlo[idx]
                feat_stla = feat_stla[idx]
                feat01 = feat01[idx]
                feat02 = feat02[idx]
                label0 = label0[idx]

            feat0 = np.column_stack((feat_stlo, feat_stla, feat01, \
                                     feat02, np.ones(feat01.size)))
            # Fake picks
            if feat0.shape[0] > 0:
                feat0[:,2] -= np.min(feat0[:,2])
                idx = np.random.choice(feat0.shape[0], size=n_fake)
                fake_picks = np.random.uniform(0, t_max, size=n_fake)
                fake_stlo = feat0[idx,0]
                fake_stla = feat0[idx,1]
                fake_pidx = np.random.choice([0, 1], size=n_fake)
                fake_labels = np.zeros(n_fake)
            else:
                fake_picks = np.random.uniform(0, t_max, size=n_fake)
                fake_sta = np.random.randint(0, n_sta, size=n_fake)
                fake_stlo = stlo[fake_sta]
                fake_stla = stla[fake_sta]
                fake_pidx = phasemap[fake_sta]
                fake_labels = np.zeros(n_fake)
            fake_feat = np.column_stack((fake_stlo, fake_stla, fake_picks, \
                                     fake_pidx, np.ones(fake_stlo.size)))

            feat1 = np.row_stack((feat0, fake_feat))
            label1 = np.concatenate((label0, fake_labels))
            if feat1.shape[0] > 0:
                feat1[:,0] = (feat1[:,0] - x_min) / \
                        (x_max - x_min)
                feat1[:,1] = (feat1[:,1] - y_min) / \
                        (y_max - y_min)

            idx = np.argsort(feat1[:,2])
            feat1 = feat1[idx,:]
            label1 = label1[idx]

            # Impose minimum time for picks
            idx = np.where(feat1[:,2] >= 0)[0]
            label1 = label1[idx]
            feat1 = feat1[idx,:]

            # Impose maximum time for picks
            idx = np.where(feat1[:,2] <= t_max)[0]
            label1 = label1[idx]
            feat1 = feat1[idx,:]

            feat1[:,2] /= t_max

            # Check that the number of picks does not exceed the max allowed
            if feat1.shape[0] >= max_picks:
                label1 = label1[:max_picks]
                feat1 = feat1[:max_picks,:]
            else:
                label1.resize(max_picks)
                feat1_ = np.zeros((max_picks, 5))
                feat1_[feat1.shape[0]:,2] = 0.0
                feat1_[:feat1.shape[0], :] = feat1
                feat1 = feat1_

            if True:
                if label1[0] == 0:
                    label1[:] = 0
                    label1[0] = 1
                else:
                    idx1 = np.where(label1 == label1[0])[0]
                    idx0 = np.where(label1 != label1[0])[0]
                    label1[idx0] = 0
                    label1[idx1] = 1
            chunk.append((feat1, label1))

        feat1 = np.array([x[0] for x in chunk])
        label1 = np.array([x[1] for x in chunk])

        out_q.put((feat1, label1))

class tt_interp:
    def __init__(self, ttfile):
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

        from scipy.interpolate import RectBivariateSpline
        self.interp_ = RectBivariateSpline(self.depths, self.dists, self.tt)

    def interp(self, dist, depth):
        return self.interp_.ev(depth, dist)


if __name__ == "__main__":
    from obspy.geodetics.base import gps2dist_azimuth
    if len(sys.argv) != 2:
        print("phaselink_dataset config_json")
        sys.exit()

    with open(sys.argv[1], "r") as f:
        params = json.load(f)

    print("Starting up...")
    phase_idx = {'P': 0, 'S': 1}
    stlo = []
    stla = []
    with open(params['station_file'], 'r') as f:
        for line in f:
            net, sta, lat, lon = line.split()
            stlo.append(float(lon))
            stla.append(float(lat))

    lat0 = (np.max(stla) + np.min(stla))*0.5
    lon0 = (np.max(stlo) + np.min(stlo))*0.5

    stations = {}
    sncl_map = {}
    count = 0
    print("Setting up initial coordinate frame")
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

    x_min = np.min(stlo)
    x_max = np.max(stlo)
    y_min = np.min(stla)
    y_max = np.max(stla)

    for key in sncl_map:
        X0, Y0 = stations[key]
        X0 = (X0 - x_min) / (x_max - x_min)
        Y0 = (Y0 - y_min) / (y_max - y_min)
        stations[key] = (X0, Y0)

    pickle.dump(stations, open(params['station_map_file'], 'wb'))
    pickle.dump(sncl_map, open(params['sncl_map_file'], 'wb'))

    in_q = mp.Queue(1000000)
    out_q = mp.Queue(1000000)

    tt_p = tt_interp(params['trav_time_p'])
    tt_s = tt_interp(params['trav_time_s'])
    proc = mp.Process(target=output_thread, args=(out_q, params))
    proc.start()

    procs = []
    for i in range(n_threads):
        print("Starting thread %d" % i)
        p = mp.Process(target=generate_phases, \
            args=(in_q, out_q, x_min, x_max, y_min, y_max, \
                  sncl_idx, stla, stlo, phasemap, tt_p, tt_s))
        p.start()
        procs.append(p)

    for i in range(n_samples//n_chunk):
        in_q.put(i)

    for i in range(n_threads):
        in_q.put(None)
    for p in procs:
        p.join()
    proc.join()
