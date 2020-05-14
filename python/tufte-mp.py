# http://www.vallis.org/salon/summary-5.html
import math
import matplotlib.pyplot as P, matplotlib.patches as MP
import numpy as N

# some settings
P.rc('figure',figsize=[5,3])    # we wish to work at the final graph size
                                # in this case 5in x 3in

P.rc('font',family='Helvetica',size=10) # work in standard sans-serif
P.rc('mathtext',fontset='stixsans')     # with math from www.stixfonts.org

# P.rc('font',family='Times New Roman',size=10)   # OR: work in standard serif
# P.rc('mathtext',fontset='stix')

P.rc('pdf',fonttype=3)          # for proper subsetting of fonts
                                # but use fonttype=42 for Illustrator editing

P.rc('axes',linewidth=0.5)      # thin axes; the default for lines is 1pt

# a pedestrian plot
fig = P.figure()

axes = P.axes([0.1,0.15,            # location of frame within figure:
               1 - 0.1  - 0.02,     # x, y, dx, dy
               1 - 0.15 - 0.02])

x = N.linspace(0,4*math.pi,100)
y = N.sin(x)

# plot takes all the usual matlab options
P.plot(x,y)

# restrict the axis to where we want it
P.axis([0,4*math.pi,-1,1])

# labels --- in all text, feel free to mix in LaTeX expressions
P.xlabel('$\phi$ (radians)')
P.ylabel('amplitude')

# do our own ticks
ticklocations = [math.pi * i for i in range(1,5)]
ticklabels = [('$%s\,\pi$' % i) for i in range(1,5)]
P.xticks(ticklocations,ticklabels)

# annotate points
dataxy = (x[40],y[40])
textxy = (0.5*math.pi,-0.5)
P.annotate('note',dataxy,textxy,
           verticalalignment='center',horizontalalignment='center',
           arrowprops={'arrowstyle': '-|>','fc': 'k'})

# add geometric shapes...
art = MP.Circle((math.pi,0),0.5,facecolor='gray',edgecolor='none')
axes.add_patch(art)
