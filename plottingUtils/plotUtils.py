import numpy as np
import matplotlib.pyplot as plt

#-- Quick function to convert human RGBA to python RGBA tuple format --#
def RGBAtoRGBAtuple(color):
    r = color[0]/255
    g = color[1]/255
    b = color[2]/255
    a = color[3]
    return (r, g, b, a)

def plotPDF(X, Y, pdf, axisRange, critDensityList, expBoundDict, plotArgs, EXPBOUNDS = True, plotName=''):
    """
    pdf: Normalized pdf
    axisRange: [xmin, xmax, ymin, ymax]
    """
    
    fig, ax = plt.subplots(figsize=(8,8))
       
    #-------------------------------------------------#
    #-- Make the contours of the critical densities --#
    #-------------------------------------------------#
    
    CS = ax.contour(X, Y, pdf, levels=critDensityList, colors=['black', 'black'], linestyles = ['--', '-']) 
    
    #---------------#
    #-- Make plot --#
    #---------------#

    # Set the aspect so that resulting figure is a square
    aspect = (axisRange[1] - axisRange[0]) / (axisRange[3] - axisRange[2])

    # Make plot
    im1 = ax.imshow(np.rot90(pdf), cmap='GnBu', aspect=aspect, extent=axisRange, interpolation='bilinear')
    
    #---------------#
    #-- Fix stlye --#
    #---------------#

    # Set plot title
    plotTitle  = plotArgs["plotTitle"]
    xAxisTitle = plotArgs["xAxisTitle"]
    yAxisTitle = plotArgs["yAxisTitle"]
                    
    
    # Set plot limits
    ax.set_xlim([axisRange[0], axisRange[1]])
    ax.set_ylim([axisRange[2], axisRange[3]])

    # Add labels to contours
    fmt = {}
    strs = [r'2 $\sigma$',r'1 $\sigma$']
    for l, s in zip(CS.levels, strs):
        fmt[l] = s
    ax.clabel(CS, CS.levels, inline=True, fmt=fmt, fontsize=10)

    # Plot colorbar and add lables
    cb = plt.colorbar(im1, orientation='vertical', fraction=0.046, pad=0.04)
    cb.set_label(plotTitle, size=20)
    cb.formatter.set_powerlimits((0, 0))

    # Set x and y axis labels
    ax.set(xlabel=xAxisTitle, ylabel=yAxisTitle)
    ax.yaxis.label.set_size(16)
    ax.xaxis.label.set_size(16)


    #---------------------------#
    #-- Add experiment bounds --#
    #---------------------------#
    if EXPBOUNDS and len(list(expBoundDict.keys())) != 0:
        
        LHC_bound = expBoundDict["LHC_mD_GeV"]
        LHC_frac  = plotArgs["LHC_frac"]
        
        plt.plot([LHC_bound, LHC_bound], 
                 [axisRange[2], axisRange[3]], 
                 color=RGBAtoRGBAtuple((191,191,191,1)))  
        plt.fill_between([axisRange[0], LHC_bound], 
                         [axisRange[2], axisRange[2]], 
                         [axisRange[3], axisRange[3]], 
                         color=RGBAtoRGBAtuple((191,191,191,0.6)))
        plt.text(LHC_bound+20, 
                 (axisRange[2] + LHC_frac*(axisRange[3] - axisRange[2])), 
                 'LHC bound', 
                 rotation=90, 
                 color=RGBAtoRGBAtuple((112,112,112,1)), 
                 fontweight='bold')
    
    #---------------# 
    #-- Save Plot --#
    #---------------# 
    if plotName is not '':
        plt.savefig(plotName, dpi=500)

    plt.show()