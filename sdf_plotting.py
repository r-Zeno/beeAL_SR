import numpy as np
from matplotlib import pyplot as plt
import os

def make_sdf(sT, sID, allID, t0, tmax, dt, sigma):
    """"
    Computes Spike Density Function from spiking data. time x neuron id
    """
    tleft= t0-3*sigma
    tright= tmax+3*sigma
    n= int((tright-tleft)/dt)
    sdfs= np.zeros((n,len(allID)))
    kwdt= 3*sigma
    i= 0
    x= np.arange(-kwdt,kwdt,dt)
    x= np.exp(-np.power(x,2)/(2*sigma*sigma))
    x= x/(sigma*np.sqrt(2.0*np.pi))*1000.0
    if sT is not None:
        for t, sid in zip(sT, sID):
            if (t > t0 and t < tmax): 
                left= int((t-tleft-kwdt)/dt)
                right= int((t-tleft+kwdt)/dt)
                if right <= n:
                    sdfs[left:right,sid]+=x
           
    return sdfs

def force_aspect(ax,aspect):
    """
    Controls aspect ratio of figs
    """
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)
    
def glo_avg(sdf: np.ndarray, n):
    """
    Returns the average sdf for glomerolus
    sdf: spike density function (time x neurons)
    n: number of neurons (of the type plotted) per glomerolus
    """
    nglo= sdf.shape[1]//n
    gsdf= np.zeros((sdf.shape[0],nglo))
    for i in range(nglo):
        gsdf[:,i]= np.mean(sdf[:,n*i:n*(i+1)],axis=1)
    return gsdf


label = "sim_20250707_193645"
pop = [
       "orn",
       "ln",
       "pn"
       ]

p = "orn"
N = 60*160
sigma_sdf = 100
dt_sdf = 1


spike_t = []
spike_ID = []

spike_t_orn = np.load(os.path.join(label, f"{p}_spike_t.npy"))
spike_ID_orn = np.load(os.path.join(label, f"{p}_spike_id.npy"))
    
trial_time = 3000 # in ms
trials = [1,2]

lsdfs_od1 = []
gsdfs_od1 = []
lsdfs_od2 = []
gsdfs_od2 = []


li = 0
li2 = 0


# computing sdf for each glom
for i in trials:
    
    if i == 1:
        left = 0
        right = left + trial_time
        
        while li < len(spike_t_orn) and spike_t_orn[li] < left: # incrementing li till at the start of the trial time window
            li += 1
        ri = li
        while ri <len(spike_t_orn) and spike_t_orn[ri] < right: # incrementing ri till at the end of the trial time window
            ri += 1
        lsdfs_od1.append(make_sdf(spike_t_orn[li:ri], spike_ID_orn[li:ri], np.arange(0,N), left, right, dt_sdf, sigma_sdf)) # originally it was "left-3*sigma_sdf" and so on for the limits,
        # i changed it beacause its already done within the make_sdf code itself. could this be a potential issue? need to check!
        gsdfs_od1.append(glo_avg(lsdfs_od1[-1],60))
        
    if i == 2:
        left2 = right + 1000
        right2 = left2 + trial_time
        
        while li2 < len(spike_t_orn) and spike_t_orn[li2] < left2:
            li2 += 1
        ri2 = li2
        while ri2 <len(spike_t_orn) and spike_t_orn[ri2] < right2:
            ri2 += 1
        # sdf for each neuron
        lsdfs_od2.append(make_sdf(spike_t_orn[li2:ri2], spike_ID_orn[li2:ri2], np.arange(0,N), left2, right2, dt_sdf, sigma_sdf))
        # mean sdf for each glom
        gsdfs_od2.append(glo_avg(lsdfs_od2[-1],60))
        

# plotting
mn= [-5, -40]
mx= [40, 200]
fig, ax = plt.subplots(1,2)

ts1 = np.transpose(gsdfs_od1[0]) # array must be inverse (neurons x time)
ax[0].imshow(ts1, vmin=mn[0], vmax=mx[0], cmap="hot")
force_aspect(ax[0],0.4)

ts2 = np.transpose(gsdfs_od2[0])
ax[1].imshow(ts2, vmin=mn[0], vmax=mx[0], cmap="hot")
force_aspect(ax[1],0.4)


x = np.arange(np.size(ts1,0))
y = np.arange(np.size(ts1,1))
X, Y = np.meshgrid(x, y, indexing= 'ij')

fig = plt.figure()
ax2 = fig.add_subplot(111, projection = '3d')

ax2.xaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))  # should set to transparent but doesnt work
ax2.yaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
ax2.zaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
ax2.xaxis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
ax2.yaxis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
ax2.zaxis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
ax2.plot_surface(X,Y,ts1, cmap = 'viridis')


# plt.savefig("response_ex.png",dpi=300)

# plt.rc('font', size=10)
# for i in range(2): # Creates two color bars
#     plt.figure()
#     ax= plt.gca()
#     # Creates a dummy image with a gradient from 0 to 99
#     ax.imshow(np.reshape(np.arange(0,100),(-1,1)),cmap='hot')
#     if i==0: 
#         yticks= np.arange(0,mx[0]+1,5) # Ticks based on the first vmin/vmax pair (mn[0], mx[0])
#     else:
#         yticks= np.arange(0,mx[1]+1,50) # Ticks based on the second vmin/vmax pair (mn[1], mx[1])
#     # Calculate tick positions in the 0-99 range of the dummy image
#     ytick_pos= (yticks-mn[i])/(mx[i]-mn[i])*100
#     ax.invert_yaxis()
#     ax.yaxis.tick_right()
#     ax.set_xticks([]) # No x-ticks needed for a vertical color bar
#     ax.set_yticks(ytick_pos)
#     ax.set_yticklabels(yticks) # Label ticks with actual data values
#     force_aspect(ax,0.05) # Make the color bar tall and thin
#     plt.savefig("colorbar"+str(i)+".png",dpi=300)
    
    