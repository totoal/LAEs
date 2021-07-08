import numpy as np
import csv
import pickle
import matplotlib.pyplot as plt
from scipy.integrate import simps

def mag_to_flux(m, w):
    c = 29979245800
    return 10**((m + 48.60) / (-2.5)) * c/w**2 * 1e8

def flux_to_mag(f, w):
    c = 29979245800
    return -2.5 * np.log10(f * w**2/c * 1e-8) - 48.60

def load_filter_tags():
    filepath = './JPAS_Transmission_Curves_20170316/minijpas.Filter.csv'
    filters_tags = []

    with open(filepath, mode='r') as csvfile:
        rdlns = csv.reader(csvfile, delimiter=',')

        next(rdlns, None)
        next(rdlns, None)

        for line in rdlns:
            filters_tags.append(line[1])

    filters_tags[0] = 'J0348'

    return filters_tags

def load_tcurves(filters_tags):
    filters_w = []
    filters_trans = []

    for tag in filters_tags:

        filename = './JPAS_Transmission_Curves_20170316/JPAS_'+tag+'.tab'
        f = open(filename, mode='r')
        lines = f.readlines()[12:]
        w = []
        trans = []

        for l in range(len(lines)):
            w.append(float(lines[l].split()[0]))
            trans.append(float(lines[l].split()[1]))

        filters_w.append(w)
        filters_trans.append(trans)


    tcurves = {
        "tag"  :  filters_tags,
        "w"    :  filters_w ,
        "t"    :  filters_trans
    }
    return tcurves

def central_wavelength(tcurves):
    w_central = []

    for fil in range(0,len(tcurves['tag'])):
        w_c = sum(np.array(tcurves['w'][fil])*np.array(tcurves['t'][fil]))     \
               / sum(tcurves['t'][fil])
        w_central.append(w_c)

    return w_central

### FWHM of a curve

def nb_fwhm(tcurves, nb_ind, give_fwhm = False):
    
    t = tcurves['t'][nb_ind]
    w = tcurves['w'][nb_ind]
    
    tmax = np.amax(t)
    
    for i in range(len(w)):
        if t[i] < tmax/2:
            pass
        else:
            w_min = w[i]
            break
            
    for i in range(len(w)):
        if t[-i] < tmax/2:
            pass
        else:
            w_max = w[-i]
            break
            
    if give_fwhm == False:
        return w_max, w_min
    if give_fwhm == True:
        return w_max-w_min

### Load no flag catalog

def load_noflag_cat(filename):
    with open(filename, mode='rb') as file:
        catalog = pickle.load(file)
    
    noflag_cat = {}

    pz        = []
    odds      = []
    mag       = []
    mag_err   = []


    for i in range(catalog['MAG'].shape[0]):
        fsum = sum(catalog['FLAGS'][i] + catalog['MFLAGS'][i])
            
        if fsum == 0:
            mag.append(catalog['MAG'][i])
            mag_err.append(catalog['ERR'][i])
            pz.append(catalog['PHOTOZ'][i])
            odds.append(catalog['PZODDS'][i])
            
    noflag_cat['MAG'] = np.array(mag)
    noflag_cat['ERR'] = np.array(mag_err)
    noflag_cat['W'] = catalog['W_CENTRAL']
    noflag_cat['FILTER'] = catalog['FILTER']
    noflag_cat['PHOTOZ'] = np.array(pz)
    noflag_cat['PZODDS'] = np.array(odds)

    return noflag_cat

## Color plot BB-NB
def plot_bbnb(mock, pm, bb_ind, nb_ind, ew0, plot_error = False):

    islae = mock['LAE']
    EW_Arr = mock['EW_Arr']

    filters_tags = load_filter_tags()
    tcurves = load_tcurves(filters_tags)
    w_central = central_wavelength(tcurves)

    w = np.array(w_central)

    c = 299792458 *100 # c in cgs
    mag = np.zeros(pm.shape)

    for i in range(pm.shape[1]):
        mag[:,i] = -2.5*np.log10(pm[:,i] * w**2/c *1e-8) - 48.60

    print(tcurves['tag'][nb_ind])
    print(tcurves['tag'][bb_ind])

    bb = mag[bb_ind,:] 
    nb = mag[nb_ind,:]
    
    bbnb = bb - nb    

    ## Color cut
    Lya_w = 1215.67
    z = w_central[nb_ind]/Lya_w - 1
    EW     = ew0 * (1+z) # A
    w_max_nb, w_min_nb = nb_fwhm(tcurves, nb_ind)
    fwhm = w_max_nb - w_min_nb
    
    print('z = ' + str(z))
    
    color_cut = 2.5*np.log10(EW/fwhm + 1)

    x_color_cut = np.linspace(15,31,100)
    y_color_cut = np.ones(100) * color_cut
    ##
    
    z_max = w_max_nb/Lya_w - 1
    z_min = w_min_nb/Lya_w - 1
    
    z_Arr = mock['redshift_Lya_Arr']
    
    isz = []
    for z in z_Arr:
        if z < z_max and z > z_min:
            isz.append(True)
        else:
            isz.append(False)
            
    bb_rightz    =    bb[np.where(isz)]
    bbnb_rightz  =  bbnb[np.where(isz)]
    islae_rightz = islae[np.where(isz)]
    
    ##### PLOT ####
    
    plt.figure(figsize=(7,5))

    plt.scatter(bb[np.where(islae == False)],  bbnb[np.where(islae == False)],
                edgecolor = 'purple', facecolor='None')
    plt.scatter(bb[np.where(islae == True )],  bbnb[np.where(islae == True )],
                marker='s', edgecolor='green', facecolor='None')
    
    plt.scatter(bb_rightz[np.where(islae_rightz == False)],
                bbnb_rightz[np.where(islae_rightz == False)],
                edgecolor = 'purple', facecolor='purple')
    plt.scatter(bb_rightz[np.where(islae_rightz == True )],
                bbnb_rightz[np.where(islae_rightz == True )],
                marker='s', edgecolor='green', facecolor='green')
    
    plt.plot(x_color_cut, y_color_cut, 'orange')

    plt.ylabel('bb-nb', size='15')
    plt.xlabel('bb'   , size='15')

    plt.ylim( (-3  ,  3 ) )
    plt.xlim( ( 19 ,  27) )

    plt.show()

if __name__ == '__main__':
    cat = load_noflag_cat('catalogDual.pkl')
    print(cat.keys())
