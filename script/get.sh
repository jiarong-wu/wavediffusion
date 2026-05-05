YEAR=2018
for i in $(seq -w 1 12); do
    echo wget "https://data-dataref.ifremer.fr/ww3/GLOBMULTI_ERA5_GLOBCUR_01/GLOB-30M/${YEAR}/FIELD_NC/LOPS_WW3-GLOB-30M_${YEAR}${i}.nc"
    wget "https://data-dataref.ifremer.fr/ww3/GLOBMULTI_ERA5_GLOBCUR_01/GLOB-30M/${YEAR}/FIELD_NC/LOPS_WW3-GLOB-30M_${YEAR}${i}.nc"
done
