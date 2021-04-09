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

# # First set up the figure, the axis, and the plot element we want to animate
# fig = plt.figure()
# ax = plt.axes(xlim=(x.min() - 0.1*(x.max() - x.min()), x.max() + 0.1*(x.max() - x.min())), ylim=(y.min() - 0.1*(y.max() - y.min()), y.max() + 0.1*(y.max() - y.min())))
# ax.set_aspect('equal')
# ax.set_axis_off()
# line, = ax.plot([], [], lw=2)

# # initialization function: plot the background of each frame
# def init():
#     line.set_data([], [])
#     return line,

# # animation function.  This is called sequentially
# def animate_whole(i):
#     global totals
#     x = totals[i].real
#     y = totals[i].imag
#     line.set_data(x, y)
#     return line,

# # Animate along the drawing rather than along fourier terms.
# def animate_line(i):
#     global totals
#     x = totals[-1, :i].real
#     y = totals[-1, :i].imag
#     line.set_data(x, y)
#     return line,

# fps = 50
# anim_line = animation.FuncAnimation(fig, animate_line, init_func=init, frames=npts, interval=1000/fps, blit=True)
# anim_line.save(f'test_{fps}fps.mp4', fps=fps, extra_args=['-vcodec', 'libx264'])
# plt.show()

# # call the animator.  blit=True means only re-draw the parts that have changed.
# anim_whole = animation.FuncAnimation(fig, animate_whole, init_func=init, frames=2*n_max+1, interval=1000/fps, blit=True)

# save the animation as an mp4.  This requires ffmpeg
# # anim_whole.save(f'basic_animation_{fps}fps.mp4', fps=fps, extra_args=['-vcodec', 'libx264'])
# plt.show()


class AnimateFourier():
    def __init__(self, fourier_terms, total_terms, drawing_coordinates, npts, animated_terms=1, fps=50):
        self.terms = fourier_terms
        self.totals = total_terms
        self.fps = fps
        self.npts = npts
        self.animated_terms = animated_terms
        self.x = drawing_coordinates.real
        self.y = drawing_coordinates.imag
        self.circles = []
        self.nterms = self.terms.shape[0]

        self.init_canvas()
        

    def init_canvas(self):
        self.fig = plt.figure()
        self.ax = plt.axes(xlim=(self.x.min() - 0.1*(self.x.max() - self.x.min()), self.x.max() + 0.1*(self.x.max() - self.x.min())), 
                           ylim=(self.y.min() - 0.1*(self.y.max() - self.y.min()), self.y.max() + 0.1*(self.y.max() - self.y.min())))
        self.ax.set_aspect('equal')
        self.ax.set_axis_off()
        self.line, = self.ax.plot([], [], lw=2)

        self.init_circles()


    def init_circles(self, cx=0, cy=0, r=1):
        for nterm in range(self.nterms):
            self.circles.append(plt.Circle((cx,cy), r, fill=False, animated=True))

        

    def init(self):
        self.line.set_data([], [])
        for circle in self.circles:
            self.ax.add_patch(circle)

        return [self.line] + self.circles

    def animate(self, i, animated_terms):
        # x = totals[0,:i].real + terms[1,:i].real
        # y = totals[0,:i].imag + terms[1,:i].imag
        x = totals[animated_terms-1,:i].real
        y = totals[animated_terms-1,:i].imag
        self.line.set_data(x,y)

        for term in range(animated_terms):
            if term != 0:
                self.circles[term].set_center((totals[term-1,i-1].real,totals[term-1,i-1].imag))
                self.circles[term].set_radius(np.abs(terms[term,0]))

        return [self.line] + self.circles

    def instantiate_animation(self):
        return animation.FuncAnimation(self.fig, self.animate, init_func=self.init, fargs=[self.animated_terms], frames=self.npts, interval=1000/self.fps, blit=True)


af = AnimateFourier(fourier_terms=terms, total_terms=totals, drawing_coordinates=f, npts=npts, animated_terms=4, fps=10)
anim = af.instantiate_animation()
#anim.save(f'test_{af.fps}fps.mp4', fps=af.fps, extra_args=['-vcodec', 'libx264'])
plt.draw()
plt.show()