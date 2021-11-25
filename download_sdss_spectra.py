import os
import sys
import numpy as np
import sys
import time


# prints a phrase which get continuously
# updated on the same line
def update_print(phrase, appendix=' ', percent=False):
    sys.stdout.write('\r')
    if percent==True:
        #sys.stdout.write(phrase + '%d%%'%appendix )
        sys.stdout.write(phrase + '%2.2f%%'%appendix )
        sys.stdout.write("\033[K")  # cleaning the terminal line
        if (appendix > 99.99): print('\n')
    if percent==False:
        sys.stdout.write(phrase + appendix )
        sys.stdout.write("\033[K")  # cleaning the terminal line
        if 'done' in appendix: print('\n')
    sys.stdout.flush()
    time.sleep(1.e-6)
    


# terminates a run after printing a sentence
def terminate(phrase):
    sys.stdout.write('\r')
    sys.stdout.write( phrase )
    sys.stdout.write('\n')
    sys.exit()





# This downloads sdss spectra, either from a single (plate,mjd,fiber) ID
# or from a list of (plate,mjd,fiber) IDs
def download_sdss_spectrum(pl=None, mj=None, fi=None, folder='', verbose=False):
    import requests
    from requests.exceptions import HTTPError

    pla = np.atleast_1d(pl)
    mjd = np.atleast_1d(mj)
    fib = np.atleast_1d(fi)

    if (len(pla)!=len(mjd)): terminate("\n\nInconsistent number of plate and mjd IDs!!\n")
    if (len(pla)!=len(fib)): terminate("\n\nInconsistent number of plate and fiber IDs!!\n")
    if (len(fib)!=len(mjd)): terminate("\n\nInconsistent number of fiber and mjd IDs!!\n")
    
    if not os.path.exists(folder): os.makedirs(folder)   # creating download folder
    os.chdir(folder)
    
    eboss_url = 'https://data.sdss.org/sas/dr14/eboss/spectro/redux/v5_10_0/spectra/lite/'  # eBOSS folder
    dr14_url  = 'https://data.sdss.org/sas/dr14/sdss/spectro/redux/26/spectra/lite/'        # non-eBOSS DR14 folder
    odd_url   = 'https://data.sdss.org/sas/dr14/sdss/spectro/redux/103/spectra/lite/'       # another SDSS folder
    what_url  = 'https://data.sdss.org/sas/dr14/sdss/spectro/redux/104/spectra/lite/'       # another SDSS folder
    
    if verbose:
        eboss_num = 0
        boss_num = 0
        other_num = 0
        what_num = 0
        not_found = 0
        already_there = 0

    cc = 0
    for i in range(len(pla)):
        pcnt = (100.*cc)/len(pla)
        update_print("Trying to download %d spectra... "%len(pla), appendix=pcnt, percent=True)
        
        spec_name = "spec-" + str(pla[i]).zfill(4) + "-%d-"%mjd[i] + str(fib[i]).zfill(4) + ".fits"
        if os.path.exists(spec_name):
            #print "\nfile %s already exists\n"%spec_name
            if verbose: already_there += 1
            continue
            
            
        if ((pla[i]!=0) & (mjd[i]!=0) & (fib[i]!=0)):
            
            sdss_fold = str(pla[i]).zfill(4) + "/"
            url_obj = eboss_url + sdss_fold + spec_name
            #os.system("wget " + url_obj)    # "direct" wget. works but does not allow to check for Error:404
            
            try:
                r = requests.get( url_obj )
                r.raise_for_status()
                #print r.status_code
                if verbose: eboss_num += 1

            except HTTPError as http_err:   # url not found at first trial, checking in other SDSS folder
                
                url_obj = dr14_url + sdss_fold + spec_name
                
                try:
                    r = requests.get( url_obj )
                    r.raise_for_status()
                    if verbose: boss_num += 1
                    
                except HTTPError as http_err:   # url not found at second trial, checking in other SDSS folder

                    url_obj = odd_url + sdss_fold + spec_name
                    try:
                        r = requests.get( url_obj )
                        r.raise_for_status()
                        if verbose: other_num += 1
                        
                    except HTTPError as http_err:   # url not found at third trial, checking in other SDSS folder
                        
                        url_obj = what_url + sdss_fold + spec_name
                        try:
                            r = requests.get( url_obj )
                            r.raise_for_status()
                            if verbose: what_num += 1
                        
                        except HTTPError as http_err:   # url not found at fourth trial
                            print("\nfile not found at fourth trial: %s\nWhere shall I look?!?"%url_obj)
                            if verbose: not_found += 1
                        
                            #--- IN CASE YOU WANT TO ADD A NEW URL TO LOOK FOR ---#
                            # url_obj = #-- new url --#
                            # try:
                            #     r = requests.get( url_obj )
                            #     r.raise_for_status()
                            
                            # except HTTPError as http_err:   # url not found at third trial
                            #     print "\nfile not found at second trial: %s\nWhere shall I look?!?"%url_obj

            open(spec_name, 'wb').write(r.content)
            cc += 1
    
    if verbose:
        print("#-- Download results of SDSS spectra --#")
        print("Expected downloads: %d"%len(pla))
        print("--------------------------")
        print("Downloaded from eBOSS: %d"%eboss_num)
        print("Downloaded from BOSS:  %d"%boss_num)
        print("Downloaded from /103/: %d"%other_num)
        print("Downloaded from /104/: %d"%what_num)
        print("Not found at all:      %d"%not_found)
        print("Previously downloaded: %d"%already_there)
        a = eboss_num + boss_num + other_num + what_num + not_found + already_there
        print("Total: %d"%a)
        print("--------------------------")