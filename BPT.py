# Instructions run the Python program: python -W ignore BPT.py
# Program to reproduce the BPT diagram using PyFits and FITS format. 

import numpy as np
import scipy as sc
from astropy.io import fits
from matplotlib import pyplot as plt
import pandas as pd

# Input data 
#change to csv from dawn archive and use pandas?? or numpy
#FITS = fits.open('http://wwwmpa.mpa-garching.mpg.de/SDSS/DR7/Data/gal_line_dr7_v5_2.fit.gz',cache=True) # Input data taken from XXX in its original format. 
# Download the data here: http://wwwmpa.mpa-garching.mpg.de/SDSS/DR7/Data/gal_line_dr7_v5_2.fit.gz
data = pd.read_csv('dja_msaexp_emission_lines_v4.4.csv')

head      = data
infheader = head.columns

# If you need 

#will need different headings for prism and grism spectra
Hbeta  = data['line_hb']          # Reading the column with Hbeta line 
OIII   = data['line_oiii_5007']       # Reading the column with OIII line
Halpha = data['line_ha']         # Reading the column with Halpha line
NII    = data['line_nii_6584']        # Reading the column with NII line


f,ax      = plt.subplots()
xx     = np.log10(  NII / Halpha )      # log10(NII/Halpha)
yy     = np.log10( OIII /  Hbeta )      # log10(OIII/Hbeta)

#bins_X, bins_Y     =  60., 60.   # Define the number of bins in X- and Y- axis
# Xmin, Xmax         = -1.2, 1.2   # Define the maximum and minimum limit in X-axis
# Ymin, Ymax         = -1.5, 1.0   # Define the maximum and minimum limit in Y-axis
# Nlevels            = 6           # Define the number of levels of isocontour


# hist,xedges,yedges = np.histogram2d(xx,yy,bins=(bins_X, bins_Y),range=[[Xmin,Xmax],[Ymin,Ymax]])
# masked             = np.ma.masked_where(hist==0, hist)
# plotting           = ax.imshow(masked.T,extent=[Xmin, Xmax, Ymin, Ymax],interpolation='nearest',origin='lower',cmap=plt.cm.gray_r)
# levels             = np.linspace(0., np.log10(masked.max()), Nlevels)[1:]
# CS                 = ax.contour(np.log10(masked.T), levels, colors='k',linewidths=1,extent=[Xmin,Xmax,Ymin,Ymax])
ax.scatter(xx,yy, alpha=0.05, color='purple')

# Kewley+01 ------------------------------------------
X = np.linspace(-1.5,0.3)
Y = (0.61/( X  - 0.47  )) + 1.19

# Schawinski+07 --------------------------------------
#X3 = np.linspace(-0.180,1.5)
#Y3 = 1.05*X3 + 0.45

# Kauffmann+03 ---------------------------------------
Xk = np.linspace(-1.5,0.)
Yk = 0.61/(Xk -0.05) + 1.3

# Regions --------------------------------------------
ax.plot(X,   Y, '-' , color='purple', lw=3, label='Kewley+01'    ) # Kewley+01 maximum starburst line
#ax.plot(X3, Y3, '-', color='black', lw=5, label='Schawinski+07') # Schawinski+07 not needed for our work
ax.plot(Xk, Yk, '--', color='purple', lw=5, label='Kauffmann+03' ) # Kauffmann+03 comes from observations, sdss galaxies based on where they lie, empircal line of starforming region
#take law 2021 line as redshift 0 and plot on its own to see where it falls

#axis labels
Nsize = 25
ax.set_xlabel(r'log([NII] $\lambda$ 6583/H$\alpha$)',fontsize=Nsize)
ax.set_ylabel(r'log([OIII] $\lambda$ 5007/H$\beta$)',fontsize=Nsize)
ax.tick_params(labelsize = Nsize)
Ymin = -1.5
Ymax = 1.0
Xmax = 1.2
Xmin = -1.2
ax.set_ylim(Ymin, Ymax)
ax.set_xlim(Xmin, Xmax)

plt.savefig('BPT_diagram.png', dpi=300)
plt.show()
