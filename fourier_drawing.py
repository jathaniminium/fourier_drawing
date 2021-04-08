import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def compute_cn(f,ns, times):
    cns = np.zeros(len(ns), dtype=np.complex128)
    dt = times[1] - times[0]
    for j,n in enumerate(ns):
        cns[j] = np.array([dt * f[i] * np.exp(n * -1j*2*np.pi*times[i]/np.max(times)) for i in range(len(times))]).sum()/np.max(times)
    return cns

def make_image(data, size=(8, 8), dpi=300):
    fig = plt.figure()
    fig.set_size_inches(size)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.set_aspect('equal')
    fig.add_axes(ax)
    ax.plot(data.real, data.imag)

#Load points
points = pd.read_csv("new_horse.csv")

# Grab x-coordinates of image
x = points.x.values

# Grab y-coordinates of image
y = points.y.values

# Combine to form complex coordinates parameterized by a "times" index array.
f = x + 1j*y

#Interpolate to more points.
n_max = 5
npts = 50*n_max + 1
times = np.linspace(0, len(f), npts)
print("Interpolating points...")
f_interp = np.interp(times, np.linspace(0,len(f), len(f)), f)

#Define what Fourier terms to compute
ns = np.linspace(-n_max, n_max, 2*n_max + 1)

#Compute Fourier term coefficients
print("Computing Fourier coefficients...")
coefficients = compute_cn(f_interp, ns, times)

#Compute magnitudes of Fourier coefficients
coeff_mags = np.abs(coefficients)

#Get indices of coefficients sorted by magnitude from largest to smallest, and sort coefficients
sorted_coeff_indices = np.argsort(coeff_mags)[::-1]
sorted_coeff = coefficients[sorted_coeff_indices]

#Compute contribution to Fourier series from each term
print("Computing contributions from each Fourier term...")
terms = np.zeros((len(ns), len(f_interp)), dtype=np.complex128)
for i,n in enumerate(ns[sorted_coeff_indices]):
    terms[i,:] = sorted_coeff[i] * np.exp(n * 1j*2*np.pi*times/np.max(times))

#Compute the cumulative contributions from each term
totals = terms.cumsum(axis=0)

print("Generating animation...")
from matplotlib import animation

# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
ax = plt.axes(xlim=(x.min() - 0.1*(x.max() - x.min()), x.max() + 0.1*(x.max() - x.min())), ylim=(y.min() - 0.1*(y.max() - y.min()), y.max() + 0.1*(y.max() - y.min())))
ax.set_aspect('equal')
ax.set_axis_off()
line, = ax.plot([], [], lw=2)

# initialization function: plot the background of each frame
def init():
    line.set_data([], [])
    return line,

# animation function.  This is called sequentially
def animate(i):
    global totals
    # plt.figtext(0.1,0.4, f"N={i+1}")
    x = totals[i].real
    y = totals[i].imag
    line.set_data(x, y)
    return line,

# call the animator.  blit=True means only re-draw the parts that have changed.
anim_whole = animation.FuncAnimation(fig, animate, init_func=init, frames=2*n_max+1, interval=100, blit=True)

# save the animation as an mp4.  This requires ffmpeg
anim_whole.save('basic_animation_20fps.mp4', fps=20, extra_args=['-vcodec', 'libx264'])
anim_whole.save('basic_animation_10fps.mp4', fps=10, extra_args=['-vcodec', 'libx264'])
anim_whole.save('basic_animation_5fps.mp4', fps=5, extra_args=['-vcodec', 'libx264'])
anim_whole.save('basic_animation_1fps.mp4', fps=1, extra_args=['-vcodec', 'libx264'])

plt.show()