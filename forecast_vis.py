# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import json

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

fname = './forecast_wave_component.json'
with open(fname, 'r') as f:
    dshb = json.load(f)

col = plt.rcParams['axes.prop_cycle'].by_key()['color']

# ---------------------------------------------------------------------------

for dd, dy in enumerate(dshb['time']):

    plt.figure(num=dy[0:10])
    plt.clf()
    plt.title(dy[0:10])

    # loop over regions and islands
    for region in [ky for ky in dshb.keys() if ky != 'time']:
        for island in dshb[region].keys():

            isl = dshb[region][island]

            for cc in range(len(isl['coastline_center'])):

                if isl['wave_component_alert_code'][cc] is None:
                    c = [0.7, 0.7, 0.7]
                elif isl['wave_component_alert_code'][cc][dd] == 2:
                    c = col[3]
                elif isl['wave_component_alert_code'][cc][dd] == 1:
                    c = col[1]
                else:
                    c = col[0]

                cst_c = isl['coastline_center'][cc]
                cst_x = [ x[0] for x in isl['coastline_coordinates'][cc] ]
                cst_y = [ y[1] for y in isl['coastline_coordinates'][cc] ]

                plt.plot(cst_x, cst_y, color=c)
                plt.plot(cst_c[0], cst_c[1], 'o', color=c)

    plt.show()












