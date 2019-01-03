# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

import numpy as np
import pandas as pd
import pickle
import netCDF4 as nc
import json
import bisect

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

def main():

    # -----------------------------------------------------------------------
    # configure

    alert_level_1 = 0.9
    alert_level_2 = 0.99

    gam_lo = 0.8
    gam_hi = 1.5

    b1_lo = 0.25
    b1_hi = 0.35

    b0_lo = 0.15
    b0_hi = 0.1

    # -----------------------------------------------------------------------
    # load forecast setup and conect to swell forecast

    with open('./forecast_setup.pickle', 'rb') as f:
        fcst_setup = pickle.load(f)

    with open('./hindcast_quantiles.pickle', 'rb') as f:
        hndcst_qtl = pickle.load(f)

    tdy = pd.Timestamp.today().strftime('%Y%m%d')
    u = 'http://nomads.ncep.noaa.gov:9090/dods/wave/mww3/' + tdy \
        + '/multi_1.ep_10m' + tdy + '_00z'
    d = nc.Dataset(u)

    fcst_length = 56 # 3-hour time steps; 56 steps = 7 days

    # -----------------------------------------------------------------------

    dshb_data = {}
    dshb_geo = {}
    slfc_crds = {}

    # loop over regions and islands
    for region in fcst_setup.keys():

        dshb_data[region] = {}

        for island in fcst_setup[region].keys():

            # if island != 'kauai':
            #     continue

            # extract relevant setup information for this island
            c = fcst_setup[region][island]
            Nc = len(c['ww3_map']) # number of coastline segments for this island

            # get forecasted swell parameters for this island
            sw = get_fcst_params(d, fcst_length, c['ww3_ij'])

            # ---------------------------------------------------------------
            # assign parameters of primary swell to coastlines

            fcst_col = pd.MultiIndex.from_product(
                [['ptl_min', 'ptl_max', 'hs', 'dp', 'tp'], range(Nc)] )
            fcst = pd.DataFrame(index=sw.index, columns=fcst_col)

            # primary swell
            for k in range(Nc):
                if c['ww3_map'][k] is not None:
                    fcst.loc[:, ('hs', k)] = \
                        sw.hs.loc[:, c['ww3_map'][k]].values
                    fcst.loc[:, ('dp', k)] = \
                        sw.dp.loc[:, c['ww3_map'][k]].values
                    fcst.loc[:, ('tp', k)] = \
                        sw.tp.loc[:, c['ww3_map'][k]].values

                    ptl = wave_breaking_potential(
                        fcst.loc[:, ('hs', k)],
                        fcst.loc[:, ('tp', k)],
                        fcst.loc[:, ('dp', k)],
                        np.array(c['normal_angles'][k])
                    )
                    fcst.loc[:, ('ptl_min', k)] = ptl.min(axis=1)
                    fcst.loc[:, ('ptl_max', k)] = ptl.max(axis=1)
                else:
                    fcst.loc[:, (slice(None), k)] = 0

            # ---------------------------------------------------------------
            # isolate parameters for daily max of wave breaking potential

            dymx_idx = sw.index.floor('D').unique()[:7]
            dymx = pd.DataFrame(index=dymx_idx, columns=fcst_col)
            for k in range(Nc):
                if c['ww3_map'][k] is not None:
                    idx_mx = fcst.ptl_max.iloc[:, k].\
                        groupby(pd.Grouper(freq='D')).idxmax()
                    for v in dymx.columns.get_level_values(0).unique():
                        dymx.loc[:, (v, k)] = fcst.loc[idx_mx, (v, k)].values
                else:
                    dymx.loc[:, (slice(None), k)] = 0

            # ---------------------------------------------------------------
            # determine alert code for each day/location

            q = hndcst_qtl[region][island]

            alert_code = pd.DataFrame(index=dymx.index,
                columns=dymx.ptl_max.columns)
            alert_code.loc[:] = 0
            alert_code[(dymx.ptl_max > q.loc[alert_level_1, :]).values] = 1
            alert_code[(dymx.ptl_max > q.loc[alert_level_2, :]).values] = 2

            # ---------------------------------------------------------------
            # calculate 2% exceedance water level range;
            # these values come back rounded to nearest 10 cm

            eta2_lo = calc_eta2(dymx.ptl_min, gam_lo, b0_lo, b1_lo)
            eta2_hi = calc_eta2(dymx.ptl_max, gam_hi, b0_hi, b1_hi)

            # ---------------------------------------------------------------
            # create time vector

            if 'time' not in dshb_data:
                dshb_data['time'] = [t.strftime('%Y-%m-%d %H:%M:%S')
                    for t in dymx.index + pd.Timedelta('12H')]

            # ---------------------------------------------------------------
            # add endpoints between each coastline segment for display

            crds = c['coordinates']
            crds_new = [[None, None] for k in range(Nc)]
            for k in range(Nc):
                prv = k - 1 if k != 0 else Nc - 1
                nxt = k + 1 if k != Nc - 1 else 0
                new_first = [
                    crds[k][0][0] - (crds[k][0][0] - crds[prv][-1][0])/2,
                    crds[k][0][1] - (crds[k][0][1] - crds[prv][-1][1])/2
                ]
                new_last = [
                    crds[k][-1][0] + (crds[nxt][0][0] - crds[k][-1][0])/2,
                    crds[k][-1][1] + (crds[nxt][0][1] - crds[k][-1][1])/2
                ]
                crds_new[k] = [new_first] + crds[k] + [new_last]

            c['coordinates'] = crds_new

            # ---------------------------------------------------------------
            # ---------------------------------------------------------------
            # aggregate information for this island into forecast dictionary

            dshb_isl = {
                'coastline_center': c['center'],
                'coastline_coordinates': c['coordinates'],
                'swell_height':\
                    ((dymx.hs*10).round().astype(int)/10).values.T.tolist(),
                'swell_period': dymx.tp.round().astype(int).values.T.tolist(),
                'swell_angle': dymx.dp.round().astype(int).values.T.tolist(),
                'swell_direction': [],
                'wave_component_alert_code': alert_code.values.T.tolist(),
                'wave_component_water_level':\
                    [ [ [lo, hi] for lo, hi in zip(elo, ehi) ]
                        for elo, ehi in
                        zip(eta2_lo.values.T.tolist(),
                            eta2_hi.values.T.tolist())
                    ],
            }

            # assign cardinal directions to swell angles
            ang = [0]
            ang.extend([a/100 for a in range(1125, 36000, 2250)])
            drn = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                   'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW', 'N']
            dshb_isl['swell_direction'] = [
                [drn[bisect.bisect(ang, jj)-1] for jj in ii]
                    for ii in dshb_isl['swell_angle']
            ]

            # for harbor coastlines, replace lists of zeros with None
            no_wave_keys = [
                'swell_height', 'swell_period',
                'swell_angle', 'swell_direction',
                'wave_component_alert_code', 'wave_component_water_level'
            ]
            no_wave_cst = [ii for ii in range(Nc) if c['ww3_map'][ii] is None]
            for key in no_wave_keys:
                for cst in no_wave_cst:
                    dshb_isl[key][cst] = None

            # --------------------------------------------------------------

            dshb_data[region][island] = dshb_isl

            # ---------------------------------------------------------------
            # ---------------------------------------------------------------
            # aggregate information into geojson format

            dshb_geo[island] = {
                'type': 'FeatureCollection',
                'features': [
                    {
                        'type': 'Feature',
                        'id': island + '_' + '{:0>2}'.format(ii),
                        'geometry': {
                            'type': 'LineString',
                            'coordinates': dshb_isl['coastline_coordinates'][ii]
                        },
                        'properties': {
                            prp: dshb_isl[prp][ii] for prp in dshb_isl if prp != 'coastline_coordinates'
                        }
                    }
                    for ii in range(len(c['center']))
                ]
            }

            # ---------------------------------------------------------------
            # ---------------------------------------------------------------

            slfc_crds[island] = c['center']

            # import sys; sys.exit()

    # -----------------------------------------------------------------------

    d.close()

    # -----------------------------------------------------------------------

    fname = './forecast_wave_component.json'
    with open(fname, 'w') as f:
        json.dump(dshb_data, f, indent=None)

    # -----------------------------------------------------------------------

    fname = './forecast_wave_component.geojson'
    with open(fname, 'w') as f:
        json.dump(dshb_geo, f, indent=None)

    # -----------------------------------------------------------------------

    fname = './sl_forecast_coordinates.json'
    with open(fname, 'w') as f:
        json.dump(slfc_crds, f, indent=None)

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

# download the relevant forecast data and put into a DataFrame
def get_fcst_params(d, N_steps, ij):

    hs = np.empty([N_steps, len(ij)])
    dp = np.empty([N_steps, len(ij)])
    tp = np.empty([N_steps, len(ij)])

    for k in range(len(ij)):
        ilon = ij[k][0]
        ilat = ij[k][1]
        hs[:, k] = d['htsgwsfc'][:N_steps, ilat, ilon].data
        dp[:, k] = d['dirpwsfc'][:N_steps, ilat, ilon].data
        tp[:, k] = d['perpwsfc'][:N_steps, ilat, ilon].data

    tidx = nc.num2date(d['time'][:N_steps], units=d['time'].units)
    col = pd.MultiIndex.from_product([ ['hs', 'dp', 'tp'] , range(len(ij)) ] )
    sw = pd.DataFrame(index=tidx, columns=col)
    sw.loc[:, 'hs'] = hs
    sw.loc[:, 'dp'] = dp
    sw.loc[:, 'tp'] = tp

    return sw

# ---------------------------------------------------------------------------

# define calculation of breaking wave potential
def wave_breaking_potential(swh, prd, drcn, nrml):

    delta_angle = drcn[:, None] - nrml[None, :]
    nrml_factor = np.cos(np.deg2rad( delta_angle ))
    nrml_factor[nrml_factor < 0] = 0

    ptl = ((swh[:, None]**2 * prd[:, None] * nrml_factor)**(2/5))

    return ptl

# ---------------------------------------------------------------------------

# define calculation of 2% exceedance water level
def calc_eta2(ptl, gam, b0, b1):

    g = 9.81

    eta2 = ptl * b1 * (np.sqrt(gam*g)/(4*np.pi))**(2/5) - b0
    eta2 *= 100 # cm

    eta2.where(eta2 > 0, other=0, inplace=True)
    eta2 = ((eta2/10).round()*10).astype(int)

    return eta2

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    main()

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------







