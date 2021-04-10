import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from svgpathtools import svg2paths

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
points = pd.read_csv("deer.csv")

# Grab x-coordinates of image
x = points.x.values

# Grab y-coordinates of image
y = points.y.values

# Combine to form complex coordinates parameterized by a "times" index array.
f = x + 1j*y

#Interpolate to more points.
n_max = 200
npts = 20*n_max + 1
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

class AnimateFourier():
    def __init__(self, fourier_terms, total_terms, drawing_coordinates, npts, animated_terms=1, n_loops=1, fps=50, speed_x=1):
        self.terms = fourier_terms
        self.totals = total_terms
        self.fps = fps
        self.speed_x = speed_x
        self.npts = npts
        self.n_loops = n_loops
        self.animated_terms = animated_terms
        self.x = drawing_coordinates.real
        self.y = drawing_coordinates.imag
        self.circles = []
        self.arrows = []
        self.nterms = self.terms.shape[0]

        self.init_canvas()
        

    def init_canvas(self):
        self.fig = plt.figure(figsize=(7,7), dpi=125)
        self.ax = plt.axes(xlim=(self.x.min() - 0.15*(self.x.max() - self.x.min()), self.x.max() + 0.15*(self.x.max() - self.x.min())), 
                           ylim=(self.y.min() - 0.15*(self.y.max() - self.y.min()), self.y.max() + 0.15*(self.y.max() - self.y.min())))   
        self.ax.set_aspect('equal')
        self.ax.set_axis_off()
        self.line, = self.ax.plot([], [], lw=2)

        self.init_circles()
        self.init_arrows()


    def init_circles(self, cx=0, cy=0, r=0, alpha=0.5):
        for nterm in range(self.nterms):
            self.circles.append(plt.Circle((cx,cy), r, alpha=alpha, fill=False, animated=True))

    def init_arrows(self, cx=0, cy=0, dx=0, dy=0, alpha=0.5):
        for nterm in range(self.nterms):
            self.arrows.append(self.ax.plot([], [], alpha=alpha, lw=1, color='r')[0])


    def init(self):
        self.line.set_data([], [])
        
        for circle in self.circles:
            self.ax.add_patch(circle)

        for arrow in self.arrows:
            # self.ax.add_patch(arrow)
            arrow.set_data([],[])

        return [self.line] + self.circles + self.arrows


    def animate(self, i, animated_terms):
        # Update outline
        if i > self.npts:
            new_i = i % self.npts
            x = np.concatenate((totals[animated_terms-1,:i].real, totals[animated_terms-1,:new_i].real))
            y = np.concatenate((totals[animated_terms-1,:i].imag, totals[animated_terms-1,:new_i].imag))
        else:
            x = totals[animated_terms-1,:i].real
            y = totals[animated_terms-1,:i].imag
        self.line.set_data(x,y)

        # Update term circles
        for term in range(animated_terms):
            if term != 0:
                if i > self.npts:
                    new_i = i % self.npts
                else:
                    new_i = i
                
                self.circles[term].set_center((self.totals[term-1,new_i-1].real,self.totals[term-1,new_i-1].imag))
                self.circles[term].set_radius(np.abs(self.terms[term,0]))

        # Update term arrows
        for term in range(animated_terms):
            if term != 0:
                if i > self.npts:
                    new_i = i % self.npts
                else:
                    new_i = i
                x1 = self.totals[term-1,new_i-1].real
                x2 = self.totals[term-1,new_i-1].real + self.terms[term,new_i-1].real
                y1 = self.totals[term-1,new_i-1].imag
                y2 = self.totals[term-1,new_i-1].imag + self.terms[term,new_i-1].imag

                self.arrows[term].set_data([x1, x2], [y1, y2])

        return [self.line] + self.arrows + self.circles

    def instantiate_animation(self):
        return animation.FuncAnimation(self.fig, self.animate, init_func=self.init, fargs=[self.animated_terms], 
                                       frames=range(0,int(self.npts*self.n_loops),self.speed_x), interval=1000/self.fps, blit=True)


def make_animation(animated_terms=2, n_loops=1, fps=15, speed_x=1, save=False):
    af = AnimateFourier(fourier_terms=terms, total_terms=totals, drawing_coordinates=f, npts=npts, animated_terms=animated_terms+1, n_loops=n_loops, fps=fps, speed_x=speed_x)
    anim = af.instantiate_animation()
    if save:
        anim.save(f'test_{animated_terms}modes_{af.fps}fps.mp4', fps=af.fps, extra_args=['-vcodec', 'libx264'])
    plt.draw()
    plt.show()
