import typing

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import typing
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


def gaussian1d(s, m, width_sigma=4, res=100):
    x = np.linspace(m-s*width_sigma, m+s*width_sigma, res)
    return x, np.exp(-0.5*((x-m)/s)**2)/s/np.sqrt(2*np.pi)

def gaussian2d(ax, C, M, width_sigma=5, res=100, debug=False):
    S0 = np.sqrt(C[0, 0]) # height
    S1 = np.sqrt(C[1, 1]) # width

    evals, evecs = np.linalg.eig(C)
    if debug: print('E', evals)
    direction = evecs[list(evals).index(max(abs(evals)))]
    if S1 < S0:
        angle = 180/np.pi * np.arctan2(direction[1], direction[0])
    else:
        angle = -180 / np.pi * np.arctan2(direction[0], direction[1])
    if debug: print('angle', angle)

    ellipses = [Ellipse(xy=np.flip(M), width=S1*n*2, height=S0*n*2, angle=angle,
                      edgecolor=color, fc='none', lw=1.5, linestyle=style) for (n, color, style) in (
        (1, 'blue', 'solid'),
        (2, '#f70', 'dashed'),
        (3, 'magenta', 'dotted')
    )]

    if debug: print(C, M)

    for e in ellipses:
        ax.add_patch(e)

    sw0 = S0 * width_sigma
    sw1 = S1 * width_sigma

    cos = lambda x: abs(np.cos(x))
    sin = lambda x: abs(np.sin(x))
    ax.set_ylim(M[0] - sw0, M[0] + sw0)
    ax.set_xlim(M[1] - sw1, M[1] + sw1)
    #ax.set_ylim(M[0] - (sw0*cos(angle)+sw1*sin(angle)), M[0] + (sw0*cos(angle)+sw1*sin(angle)))
    #ax.set_xlim(M[1] - (-sw1*cos(angle)+sw0*sin(angle)), M[1] + (-sw1*cos(angle)+sw0*sin(angle)))

    #


PANN = r'$p_{\mathrm{ann}}$ ($10^{-7}\mathrm{m}^3\mathrm{s}^{-1}\mathrm{kg}^{-1}$)'

def triplot(labels: typing.Sequence[str], scales, means: np.ndarray, covariances: np.ndarray, *hard_limits, debug=False):
    dim = len(labels) #covariances.shape[0]
    #if len(labels) != dim: raise ValueError("Wrong number of labels %s, should be %s" % (len(labels), dim))

    fig: plt.Figure = plt.figure()
    tickfont = {'family': 'serif',
            'weight': 'normal',
            'size': 9}
    plt.rc('font', **tickfont)
    font = {'family': 'serif',
            'weight': 'normal',
            'size': 10}

    axes: np.ndarray[list[plt.Subplot]] = fig.subplots(dim, dim)
    fig.subplots_adjust(left=None, bottom=0.15, right=None, top=0.98, wspace=0, hspace=0)

    #fig.suptitle('Parameter Covariances')

    for i in range(dim):
        for j in range(i+1):
            if debug: print(i,j)
            ax: plt.Axes = axes[i,j]
            ax.set_aspect('auto', 'box')
            ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.5f'))
            ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.5f'))

            C = np.array([[covariances[i, i]*10**scales[i]*10**scales[i],
                           covariances[i, j]*10**scales[i]*10**scales[j]],
                          [covariances[j, i]*10**scales[i]*10**scales[j],
                           covariances[j, j]*10**scales[j]*10**scales[j]]])
            M = np.array((means[i]* 10**scales[i], means[j]* 10**scales[j]))
            S0 = np.sqrt(C[0,0])
            S1 = np.sqrt(C[1,1])

            # first column and last row: set labels and tick spacings
            if j == 0:
                ax.set_ylabel(labels[i], **font)
                ax.set_yticks(np.linspace(0, 2* 3* S0, 4)-3*S0+M[0])
                ax.set_yticklabels(np.round(ax.get_yticks(), 6), rotation=0, ha="right")
            else:
                plt.setp(ax.get_yticklabels(), visible=False)
            if dim-i == 1:
                ax.set_xlabel(labels[j], **font)
                ax.set_xticks(np.linspace(0, 2 * 3 * S1, 6) - 3*S1 + M[1])
                ax.set_xticklabels(np.round(ax.get_xticks(), 6), rotation=45, ha="right")
            else:
                plt.setp(ax.get_xticklabels(), visible=False)



            # hypotenuse of triangle: 1d gaussians
            if j == i:
                x, y = gaussian1d(S0, M[0])
                y = y/max(y)
                ax.plot(x, y)
                ax.set_ylim(0, 1.2)
                ax.set_yticks([])
                if PANN == labels[i]:
                    ax.set_xlim(0)

            # others
            else:
                if not (j == 0 or dim-i == 1):
                    ax.set_xticks([])
                    ax.set_yticks([])
                gaussian2d(ax, C, M)
                if PANN == labels[j]:
                    ax.set_xlim(0., None)
                if PANN == labels[i]:
                    ax.set_ylim(0., None)

            #limy = limits.get(labels[i] )
            #limx = limits.get(labels[j])
            #ax.set_xlim()
        for j in range(j+1, dim):
            axes[i,j].axis('off')

if __name__ == "__main__":

    SCALES = [0, 0, 9, 0, 0, 0, 7]
    FID = np.array([0.1197, 0.0222, 2.196e-9, 0.9655, 0.06, 0.010409, 0])
    LABELS = [r'$\Omega_{DM} h^2$', r'$\Omega_b h^2$', r'$10^{9}A_s$', r'$n_s$', r'$\tau_{reio}$', r'$\theta_s$', PANN]

    COV = np.array([[ 2.47001346e-07,-2.97845797e-10,-8.18061556e-15,-5.52273721e-07,
  -2.21746109e-06,-1.21338986e-10,-2.13802800e-12],
 [-2.97845797e-10, 7.21294199e-10, 6.76308243e-17,-2.32036146e-08,
   1.03829820e-08,-2.87788242e-12, 4.39861855e-14],
 [-8.18061556e-15, 6.76308243e-17, 3.02396605e-22, 1.47808940e-14,
   7.94149379e-14, 3.81554984e-18, 8.07361101e-20],
 [-5.52273721e-07,-2.32036146e-08, 1.47808940e-14, 2.92442667e-06,
   4.79205507e-06, 4.65995335e-10, 2.41363118e-12],
 [-2.21746109e-06, 1.03829820e-08, 7.94149379e-14, 4.79205507e-06,
   2.13279742e-05, 1.07041799e-09, 2.06802584e-11],
 [-1.21338986e-10,-2.87788242e-12, 3.81554984e-18, 4.65995335e-10,
   1.07041799e-09, 3.16657431e-13, 7.94962807e-16],
 [-2.13802800e-12, 4.39861855e-14, 8.07361101e-20, 2.41363118e-12,
   2.06802584e-11, 7.94962807e-16, 4.80898412e-16]])

    COV2 = np.array([[ 2.47293942e-07,-3.05765884e-10,-8.19252807e-15,-5.52558470e-07,
  -2.22048548e-06,-1.21451155e-10,-1.73258203e-12],
 [-3.05765884e-10, 7.21335060e-10, 6.79718610e-17,-2.31876905e-08,
   1.04718770e-08,-2.87375017e-12, 1.64994691e-14],
 [-8.19252807e-15, 6.79718610e-17, 3.02869154e-22, 1.47915746e-14,
   7.95348488e-14, 3.81990796e-18, 6.12812197e-20],
 [-5.52558470e-07,-2.31876905e-08, 1.47915746e-14, 2.92428603e-06,
   4.79464401e-06, 4.66050708e-10, 3.25307538e-12],
 [-2.22048548e-06, 1.04718770e-08, 7.95348488e-14, 4.79464401e-06,
   2.13583617e-05, 1.07151098e-09, 1.62681900e-11],
 [-1.21451155e-10,-2.87375017e-12, 3.81990796e-18, 4.66050708e-10,
   1.07151098e-09, 3.16692200e-13, 7.98861097e-16],
 [-1.73258203e-12, 1.64994691e-14, 6.12812197e-20, 3.25307538e-12,
   1.62681900e-11, 7.98861097e-16, 4.70672490e-16]])

    COV3 = np.array([[ 2.37160937e-07,-4.04817229e-10,-7.85839340e-15,-5.22608148e-07,
  -2.12715696e-06,-1.15583085e-10, 4.73633446e-11],
 [-4.04817229e-10, 7.21542864e-10, 7.18046710e-17,-2.28679855e-08,
   1.15393512e-08,-2.82582367e-12, 1.35740375e-11],
 [-7.85839340e-15, 7.18046710e-17, 2.92123906e-22, 1.38167803e-14,
   7.65319878e-14, 3.62185591e-18, 4.74223572e-18],
 [-5.22608148e-07,-2.28679855e-08, 1.38167803e-14, 2.83635137e-06,
   4.52229232e-06, 4.48490731e-10, 1.54784148e-10],
 [-2.12715696e-06, 1.15393512e-08, 7.65319878e-14, 4.52229232e-06,
   2.05191914e-05, 1.01621755e-09, 1.28782630e-09],
 [-1.15583085e-10,-2.82582367e-12, 3.62185591e-18, 4.48490731e-10,
   1.01621755e-09, 3.13369811e-13,-1.32747294e-13],
 [ 4.73633446e-11, 1.35740375e-11, 4.74223572e-18, 1.54784148e-10,
   1.28782630e-09,-1.32747294e-13, 1.45370851e-10]])


    COV4 = np.array([[ 2.47020555e-07,-2.99295978e-10,-8.18263428e-15,-5.52284117e-07,
  -2.21794836e-06,-1.21356838e-10, 1.00702093e-12],
 [-2.99295978e-10, 7.21133500e-10, 6.77695425e-17,-2.31919775e-08,
   1.04200004e-08,-2.87548597e-12,-1.37566725e-14],
 [-8.18263428e-15, 6.77695425e-17, 3.02501051e-22, 1.47801598e-14,
   7.94401913e-14, 3.81625753e-18,-3.66959201e-20],
 [-5.52284117e-07,-2.31919775e-08, 1.47801598e-14, 2.92390253e-06,
   4.79171318e-06, 4.65933568e-10,-1.88915943e-12],
 [-2.21794836e-06, 1.04200004e-08, 7.94401913e-14, 4.79171318e-06,
   2.13340250e-05, 1.07057287e-09,-9.76846020e-12],
 [-1.21356838e-10,-2.87548597e-12, 3.81625753e-18, 4.65933568e-10,
   1.07057287e-09, 3.16655548e-13,-4.50697346e-16],
 [ 1.00702093e-12,-1.37566725e-14,-3.66959201e-20,-1.88915943e-12,
  -9.76846020e-12,-4.50697346e-16, 4.61545167e-16]])

    triplot(LABELS, SCALES, FID, COV, [(0, 1), (0, 2)])  # curvature -1
    #triplot(LABELS, SCALES, FID, COV2, [(0, 1), (0, 2)])
    #triplot(LABELS, SCALES, FID, COV3, [(0, 1), (0, 2)])
    triplot(LABELS, SCALES, FID, COV4, [(0, 1), (0, 2)])  # flat
    plt.show()