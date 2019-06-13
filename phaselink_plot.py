#! /bin/env python
import numpy as np
import sys
import glob
import obspy
import pylab as plt
from obspy.geodetics.base import gps2dist_azimuth
import pickle

max_resid = 1.5

class Arrival():
    def __init__(self, net=None, sta=None, time=None, phase=None,
        dist=None, resid=None):
        self.net = net
        self.sta = sta
        self.time = time
        self.phase = phase
        self.dist = dist
        self.resid = resid

class Event():
    def __init__(self, arrivals = None):
        if arrivals is not None:
            self.arrivals = arrivals
        else:
            self.arrivals = []

def read_nlloc_output(fname):
    count = 0
    arrivals = []
    with open(fname, 'r') as f:
        for line in f:
            temp = line.split()
            if count < 17:
                if temp[0] == "GEOGRAPHIC":
                    origin_time = obspy.UTCDateTime("%s-%s-%sT%s:%s:%s" % \
                        (temp[2], temp[3], temp[4], temp[5], temp[6], temp[7]))
                count += 1
                continue
            if temp[0] == "END_PHASE":
                break
            sta = temp[0]
            phase = temp[4]
            ymd = temp[6]
            hm = temp[7]
            sec = temp[8]

            resid = float(temp[16])
            if abs(resid) >= max_resid:
                continue

            year = ymd[:4]
            month = ymd[4:6]
            day = ymd[6:8]
            hour = hm[:2]
            mins = hm[2:]
            time = obspy.UTCDateTime("%s-%s-%sT%s:%s:%s" % \
                (year, month, day, hour, mins, sec))

            dist = float(temp[-6])

            arrivals.append(Arrival(sta=sta, phase=phase, time=time,
                dist=dist, resid=resid))

    return arrivals, origin_time

def get_unassociated_trigs(origin_time, triggers):
    t_start = origin_time - obspy.UTCDateTime(0) - 20.0
    t_stop = t_start + 40.0
    idx = np.where((triggers >= t_start) & (triggers < t_stop))[0]
    trigs = {}
    for x in idx:
        if trig_meta[x][1] not in trigs:
            trigs[trig_meta[x][1]] = []
        trigs[trig_meta[x][1]].append(trig_meta[x][3])
    return trigs

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("plot_detections.py file_path station_list wf_path")
        sys.exit()
    hypo_path = sys.argv[1]
    station_list = sys.argv[2]
    wf_path = sys.argv[3]

    triggers = []
    trig_meta = []
    with open("anza_gpd.output", 'r') as f:
        for line in f:
            net, sta, phase, time, prob, dur = line.split()
            if float(prob) < 0.5:
                continue
            if float(dur) < 0.5:
                continue
            triggers.append(obspy.UTCDateTime(time) - obspy.UTCDateTime(0))
            trig_meta.append((net, sta, phase, obspy.UTCDateTime(time)))
    idx = np.argsort(triggers)
    triggers = np.array([triggers[x] for x in idx])
    trig_meta = [trig_meta[x] for x in idx]

    stations = {}
    with open(station_list, 'r') as f:
        for line in f:
            net, sta, lat, lon, elev = line.split()
            stations[sta] = (float(lat), float(lon))

    for file_name in glob.glob("%s.*.*.*.*.hyp" % hypo_path):
        arrivals, origin_time = read_nlloc_output(file_name)
        trigs = get_unassociated_trigs(origin_time, triggers)

        picks = []
        for arrival in arrivals:
            sta = arrival.sta
            dist = arrival.dist
            picks.append((arrival.time, arrival.phase, \
                arrival.sta, arrival.dist, arrival.resid))
        picks = sorted(picks, key=lambda x: x[3])
        for pick in picks:
            print(pick[2], pick[1], pick[3], pick[4])

        sta_order = []
        for i in range(len(picks)):
            if picks[i][2] not in sta_order:
                sta_order.append(picks[i][2])
        pick_dict = {}
        for pick in picks:
            time, phase, sta, dist, resid = pick
            if sta not in pick_dict:
                pick_dict[sta] = [(time, phase)]
            else:
                pick_dict[sta].append((time, phase))
        picks = pick_dict

        fig, ax = plt.subplots(1,1,figsize=(12,12))
        count = 0
        for sta in sta_order:
            st = obspy.read("%s/%s/%s/*.%s.*" % \
                (wf_path, origin_time.year, origin_time.julday, sta),
                starttime=origin_time-20., endtime=origin_time+40)
            st.detrend()
            st.filter(type='bandpass', freqmin=2.0, freqmax=15)
            for tr in st:
                ax.plot(np.arange(tr.data.size)*tr.stats.delta - 20, \
                        tr.data/np.max(tr.data) + count, c='k', lw=1)
                ax.text(65, count, sta)
                if sta in trigs:
                    for pick in trigs[sta]:
                        tr_slice = tr.slice(starttime=pick, endtime=pick+1.0)
                        ax.plot(np.arange(tr_slice.data.size)*tr.stats.delta + (pick - origin_time), tr_slice.data/np.max(tr.data) + count, c='lime', lw=1)

                for pick, phase in picks[sta]:
                    if phase == 'P':
                        color = 'r'
                    else:
                        color = 'b'
                    tr_slice = tr.slice(starttime=pick, endtime=pick+1.0)
                    ax.plot(np.arange(tr_slice.data.size)*tr.stats.delta + (pick - origin_time), tr_slice.data/np.max(tr.data) + count, c=color, lw=1)
                count += 1
        plt.show()
        print()
