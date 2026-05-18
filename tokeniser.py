import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from astropy.table import Table
from astropy.io import fits
from astropy import units as u

import torch as t


#------------------------------------------CONFIGURATION----------------------------------------------------#

#can test for one or two spectrums to test that its wokring for the big batch 

data = Table.read(f'dja_msaexp_emission_lines_v4.4.prism_spectra.fits')

#for flux values current dims after running this section amount of spectrum,473
f = data['flux']
f = np.array(f)
#f = f[:, :amount of spectrum] #f values were matrix of all spectrums so we only wanted to take 100 on the columns of the matrix
f = f.T #this transposes the matrix to make batch the first dimension for using pytorch ltr
#dimension = rows,columns = B,L = (amount of spectrum,473) where B is amount of spectra/batches and L is observation length of flux array

#for wavelengths
w = data['wave'] #w is not a matrix bc maybe same wavelength values for the grating
#dims = L = (473)

f /= w**2 #units
f_normalised = f / np.nanmean(f,axis=1,keepdims=True) # flux_conversion(data['flux']) #is this the correct normalising, in paper when they normalise they subtract mean and divide by std
#this is now changed because before it was a vector now is a matrix, so we want ot make sure we are doing the division just to the rows which is axis =1 
#dimensions of f_normalised = amount of spectrum,473

# --------------- making the big for loop ------------------ # 

####creating overlapping patches 
patch_size = 20
overlap = 10
L = f_normalised.shape[1] #to get the 473 for length
B = f_normalised.shape[0] #to get the amount of spectrum for length

step_size = patch_size - overlap 

#this does the for loop but faster 
pad_length = patch_size - L % step_size
f_normalised = np.append(f_normalised, np.ones((B,pad_length))*np.nan, axis=1) #B, pad_length makes it pad only to the columns (473), axis = 1 applys the np.append along the L dimension which is along the 473


x_t = sliding_window_view(f_normalised, patch_size, axis = 1)[:,::step_size] #(amount of spectrum,48,20) amount of spectrum (34000) is B, 48 is how many patches w fit into the thing which is t and 20 is patch_size
X = np.concatenate([np.nanmean(x_t,axis=2, keepdims = True), np.nanstd(x_t,axis=2, keepdims = True), x_t], axis=2) #adding mean and std to create matrix, axis = 2 adds to flux values

#print(f_normalised.shape, x_t.shape, X.shape)

#-------------- need to build the binary mask -----------------#

#revisit below we might have done this wrong
#this is so that for x_t vectors that include 50% nans or more, the model knows these are invalid

threshold = 0.75 #set a threshold will change later but for now this allows the mask to mark a patch as invalid if it has over this amount of nans

M = (np.isnan(X).sum(axis=2) / patch_size) >= threshold #the mask criteria put together, check the axis here ltr
 

#--------------- builting the wavelength embedding ------------------#

#need to pad wavelength array without adding nans

lam = np.append(w, w[-1] + np.arange(1, pad_length+1)*(w[-1] - w[-2])) #this linearly extrapolates wavelength values at the end of the array to meet the required pad_length

x_lam = sliding_window_view(lam, patch_size)[::step_size] #the patches for the wavelength array slightly overlapping

W_lam = np.mean(x_lam, axis=1, keepdims = True) #a vector of the means from the patches

#------------- sinousiodal wavelength encoder ----------------------#

# create p that is t long and d wide (define abritrary) each element in p must satisfy c onfitions in paper       
#something gone wrong w xshape
t = X.shape[1] #currently just the number of patches 
D_emb = patch_size + 2 

P_global = np.zeros((B,t,D_emb)) #empty matrix of 0's of dimensions B t and D_emb


omegas = 10000 ** (- 2 * np.arange(D_emb // 2) / D_emb) #what is written in paper using int division


product = W_lam * omegas # this is from paper! added 10000, removed for now but why should we add it in again


# applied to each element
sines = np.sin(product)
coses = np.cos(product) 


even_mask = np.arange(D_emb) % 2 == 0 # np.arange(D_emb) makes (0, 1, 2,. .. , D_emb-1),  % 2 == 0 means make it True if it is divisible by two, otherwise it is False

#print(even_mask)

# : means just do this for all of the B's and all of the T's
P_global[:, :,  even_mask] = sines[np.newaxis,...] #adds a batch dimension to sines and coses to shapes match up, numpy new axis adds a new dimension 1 to make it consistent because we are assigning it to something with a batch dimension so it needs to have something w that shape
P_global[:, :, ~even_mask] = coses[np.newaxis,...]

#print(P)

#print(P.shape, X.shape)





#-----------------creating Z'-------------------------#

#create Z which is X after embedding

# #Z_x = 

# #weights?

# #create Z which is combination of Z_x and P
# Z = Z_x + P #do i need to use np to do an append for a matrix or a concatenate?





#------------------------------------------TRANSFORMER BLOCK----------------------------------------------------#
#create validity mask to show that the token is a non padded token when M = 1, this might be different to what we made before
#3 transformer layers?

# def ValidityMask(t):
#     threshold = 0.75
#     M = (np.isnan(X).sum(axis=2) / patch_size) >= threshold #need to make sure this is performing the mask on patches t in batches B which i dont think it is
    
#     if M == True
#     return M = 1 

# return M = 0

#additive attention mask 

#A = 
#create for binary validity signal, what is this applying to, is it to P or Z or X?

#3 attention heads? 

#Z_prime = Z + MHA(LayerNorm(Z), A) #multihead attention (MHSA is multi head self attention) this is for the preactivation risidual structure for each transformer layer
#Z_out = Z_prime + FFN(LayerNorm(Z_prime)) #this is for the same preactivation but adds in a feed forward network (FFN)



#------------------------------------------LOCAL POSITIONAL EMBEDDING-----------------------------------------------------#
#augment a value tensor V with a depthwise convolution

#how to create V?
#V = 

#create Conv1D or is it imported
#Conv1D is a grouped convolution with kernel size 3 and group count = dh
#group_count = d_h
#P_local = Conv1D(V)


#i think this section in the paper is written with little detail
# MHA(Z) = np.concatenate()



#------------------------------------------RECONSTRUCTION LOSS -----------------------------------------------------#











