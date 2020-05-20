#!/home/zross/bin/python 

import numpy as np
import sys
import glob
import obspy
import pylab as plt
import json
import random

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

def get_unassociated_trigs(origin_time, triggers, trig_meta):
    t_start = origin_time - obspy.UTCDateTime(0) - 60.0
    t_stop = t_start + 120.
    idx = np.where((triggers >= t_start) & (triggers < t_stop))[0]
    trigs = {}
    for x in idx:
        if trig_meta[x][1] not in trigs:
            trigs[trig_meta[x][1]] = []
        trigs[trig_meta[x][1]].append((trig_meta[x][3], trig_meta[x][4]))
    return trigs

def plot_seismicity(catalog, params):
    import pandas as pd

    print('Reading fault file in GMT format, please wait...')

    # list to store fault segments
    faults = []

    # preallocate to track fault pts within segment
    maxpts = 1600000 # based on number of lines in file
    flats = np.zeros(maxpts)
    flons = np.zeros(maxpts)
    fsegs = np.zeros(maxpts,dtype='int')
    nn = -1
    nseg=-1

    # loop over lines
    with open(params['fault_file']) as f:
        for line in f:

            # header line that gives number of points in segment
            if line.startswith('Pline'):
                nseg+=1

            # fault point line
            elif line.startswith('-1'):
                nn+=1
                lineS = line.split()
                flons[nn]=float(lineS[0])
                flats[nn]=float(lineS[1])
                fsegs[nn]=nseg

    # covert to dataframe
    fault_df = pd.DataFrame()
    fault_df['flon']=flons[:nn+1]
    fault_df['flat']=flats[:nn+1]
    fault_df['fseg']=fsegs[:nn+1]
    print('Done, {:} faults read'.format(nseg+1))

    from mpl_toolkits.basemap import Basemap, shiftgrid, cm
    fig = plt.figure(figsize=(10,10))
    ax = plt.gca()
    lat0, lat1 = params['lat_min'], params['lat_max']
    clat = (lat0+lat1)/2.
    lon0, lon1 = params['lon_min'], params['lon_max']
    clon = (lon0+lon1)/2.

    proj = 'merc'
    epsg = 4269
    m = Basemap(llcrnrlon=lon0,llcrnrlat=lat0,urcrnrlon=lon1,urcrnrlat=lat1,
                resolution='h',projection=proj,lat_0=clat,lon_0=clon, ax=ax,
                epsg=epsg)
    m.drawcoastlines()
    m.fillcontinents(color='white', lake_color='paleturquoise')
    m.drawparallels(np.arange(32, 38, 1.), labels=[1,0,0,1])
    m.drawmeridians(np.arange(-120, -114, 1.), labels=[1,0,0,1])
    m.drawmapboundary(fill_color='paleturquoise')


    xpixels = 5000
    service = 'World_Shaded_Relief'
    #m.arcgisimage(service=service, xpixels = xpixels, verbose= False)

    # plot faults
    ifaults = (fault_df.flat >= lat0)&(fault_df.flat <= lat1) & (
                fault_df.flon >= lon0)&(fault_df.flon <= lon1)
    for g, v in fault_df[ifaults].groupby('fseg'):
        m.plot(v.flon.values,v.flat.values,'-k',lw=1.0,latlon=True)

    lon = []
    lat = []
    for event in cat:
        lon.append(event.origins[0].longitude)
        lat.append(event.origins[0].latitude)
    #with open("datasets/cahuilla_sum.nll", 'r') as f:
    #    for line in f:
    #        temp = line.split()
    #        lon.append(float(temp[11]))
    #        lat.append(float(temp[9]))

    m.scatter(lon, lat, 0.5, marker='o', color='r', latlon=True, zorder=10)
    stla = []
    stlo = []
    with open(params["station_file"], 'r') as f:
        for line in f:
            temp = line.split()
            stla.append(float(temp[2]))
            stlo.append(float(temp[3]))
    m.scatter(stlo, stla, 50, marker='^', color='blue', latlon=True, zorder=10)
    plt.tight_layout()
    plt.savefig("detection_map.png", dpi=320)
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("phaselink_plot.py control_file.json")
        sys.exit()


    with open(sys.argv[1], "r") as f:
        params = json.load(f)

    triggers = []
    trig_meta = []
    if params['plot_unassociated']:
        print("Reading unassociated triggers...")
        with open(params['gpd_file'], 'r') as f:
            for line in f:
                net, sta, phase, time, prob, dur = line.split()
                if float(prob) < params['pr_min'] or \
                   float(dur) < params['trig_dur_min']:
                    continue
                    trig_type = 0
                else:
                    trig_type = 1
                triggers.append(obspy.UTCDateTime(time) - obspy.UTCDateTime(0))
                trig_meta.append((net, sta, phase, obspy.UTCDateTime(time),
                    trig_type))
        idx = np.argsort(triggers)
        triggers = np.array([triggers[x] for x in idx])
        trig_meta = [trig_meta[x] for x in idx]

    print("Now building catalog")

    #nll_summary_file = "%s/%s" % \
    #    (params['nlloc_loc_path'], params['nlloc_sum_file'])
    #cat = obspy.io.nlloc.core.read_nlloc_hyp(nll_summary_file)
    nll_files = glob.glob("%s/*.*.*.*.*.hyp" % params['nlloc_loc_path'])
    cat = obspy.Catalog()
    for fname in nll_files:
        try:
            cat += obspy.read_events(fname)
        except:
            continue
    random.shuffle(nll_files)

    for event in cat:
        print(event.preferred_origin().time)
    print(cat)
    print()

    if params['plot_seismicity']:
        plot_seismicity(cat, params)


    for fname in nll_files:
        cat = obspy.read_events(fname)
        event = cat[0]
        origin = event.preferred_origin()
        origin_time = origin.time
        print(event)
        print(origin)

        if params['plot_unassociated']:
            trigs = get_unassociated_trigs(origin_time, triggers, trig_meta)

        # Build id_map for join between arrivals and picks
        picks = {}
        sta_order = []
        dist_count = 0
        for arrival in origin.arrivals:
            pick = arrival.pick_id.get_referred_object()
            sta = pick.waveform_id.station_code
            phase = arrival.phase
            time = pick.time
            #if arrival.distance <= params['dist_cutoff_radius']:
            #    dist_count += 1
            if abs(arrival.time_residual) > params['max_t_resid']:
                flag = 1
            else:
                flag = 0
            if sta not in picks:
                picks[sta] = [(time, phase, flag)]
                sta_order.append(sta)
            else:
                picks[sta].append((time, phase, flag))

        #if dist_count < params['dist_cutoff_n_min']:
        #    print("Skipping event, only %d phases within radius %.2f" % \
        #        (dist_count, params['dist_cutoff_radius']))
        #    continue

        # Plot results
        fig, ax = plt.subplots(1,1,figsize=(30,30))
        colors = {0: 'lime', 1: 'yellow'}
        count = 0
        for sta in sta_order:

            st = obspy.read("%s/%04d/%03d/*.%s.*" % \
                (params['wf_path'], origin_time.year, origin_time.julday, sta),
                starttime=origin_time-60, endtime=origin_time+60)
            st.detrend()
            st.filter(type='bandpass', freqmin=3.0, freqmax=20)
            for tr in st:
                ax.plot(np.arange(tr.data.size)*tr.stats.delta, \
                        tr.data/np.max(tr.data) + count, c='k', lw=1)
                ax.text(125, count, sta)
                if params['plot_unassociated']:
                    if sta in trigs:
                        for pick, t_type in trigs[sta]:
                            #tr_slice = tr.slice(starttime=pick,
                            #                    endtime=pick+1.0)
                            #ax.plot(np.arange(tr_slice.data.size) \
                            #    * tr.stats.delta + (pick - origin_time) + 60.,
                            #    tr_slice.data/np.max(tr.data) + count,
                            #    c=colors[t_type], lw=1)
                            ax.plot(pick-tr.stats.starttime, 0,
                                    marker="|", c=colors[t_type])

                for pick, phase, flag in picks[sta]:
                    if phase == 'P':
                        color = 'r'
                    else:
                        color = 'b'
                    #if flag:
                    #    color = 'limegreen'
                    ax.plot([pick-tr.stats.starttime, pick-tr.stats.starttime], [count-0.75, count+0.75], c=color)
                count += 1
        plt.show()
        print()
