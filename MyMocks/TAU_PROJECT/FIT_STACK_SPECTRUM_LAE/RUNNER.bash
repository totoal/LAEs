#sn_min = float( sys.argv[1] ) #5.5
#
#mag_sdss_g_min = float( sys.argv[2] )
#mag_sdss_g_max = float( sys.argv[3] )
#
#plae_classification_min = float( sys.argv[4] )
#combined_plae_min       = float( sys.argv[5] )

for mag_min in 22.5 23.0
do
for mag_max in 24.0 24.5 25.0 26.
do
for SN in 6.5 7.0 
do
    #ipython MCMC_STACK_LAE.py $SN $mag_min $mag_max 0.85 70. &
    #ipython MCMC_STACK_LAE.py $SN $mag_min $mag_max 0.7  60. &
    #ipython MCMC_STACK_LAE.py $SN $mag_min $mag_max 0.6  50. &
    ipython MCMC_STACK_LAE.py $SN $mag_min $mag_max 0.5  10. &
done
done
done













