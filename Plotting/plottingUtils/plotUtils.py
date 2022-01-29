import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# Set default fonts to Computer Modern (LaTeX)
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


#-- Better formatting for colorbar --#
def cbFMT(x, pos):
    a, b = '{:.2e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)

#-- Quick function to convert human RGBA to python RGBA tuple format --#
def RGBAtoRGBAtuple(color):
    r = color[0]/255
    g = color[1]/255
    b = color[2]/255
    a = color[3]
    return (r, g, b, a)

def sciNotationString(x, p='%.2E'):
    # --------------------------------------------------------------------------------------
    # Takes a number x and returns this number expressed in scientific notation up to precision p
    #
    # Inputs:  x = Number
    #          p = Desired precision
    #
    # Outputs: x as a formatted string in scientific notation for plotting
    # --------------------------------------------------------------------------------------

    s = p% x
    snew = r'$'+s[0:2+int(p[2])]+r' \times 10^{'+s[-3:]+'}$'

    return snew

def plotZcmap(plotArgs, X_cmap, Y_cmap, Z_cmap, contourDict=None, textDict=None):
    
    #------------------#
    #-- Check inputs --#
    #------------------#
    
    # Check that mandatory plotArgs are included
    mandatoryPlotArgs = ['plotTitle', 'zAxisTitle', 'xAxisTitle', 'yAxisTitle']
    if np.all([x in list(plotArgs.keys()) for x in mandatoryPlotArgs]):
        plotTitle  = plotArgs['plotTitle']
        zAxisTitle = plotArgs['zAxisTitle']
        xAxisTitle = plotArgs['xAxisTitle']
        yAxisTitle = plotArgs['yAxisTitle']
    else:
        print("Error: Missing Mandatory plot arguments, please make sure the following are in plotArgs.")
        print("       mandatoryPlotArgs = ['plotTitle', 'zAxisTitle', 'xAxisTitle', 'yAxisTitle']")
        return
    
    # Set size of axes and titles if not specified by user
    if 'axisLabelSize' in list(plotArgs.keys()):
        axisLabelSize  = plotArgs['axisLabelSize']
    else:
        axisLabelSize  = 22
    if 'titleLabelSize' in list(plotArgs.keys()):
        titleLabelSize = plotArgs['titleLabelSize']
    else:
        titleLabelSize = 24
    if 'axisTickSize' in list(plotArgs.keys()):
        axisTickSize   = plotArgs['axisTickSize']
    else:
        axisTickSize   = 16
    
    
    # Get axisRange
    if 'axisRange' in list(plotArgs.keys()):
        axisRange = plotArgs['axisRange'] # Format [xmin, xmax, ymin, ymax]
    else:
        axisRange = [X_cmap.min(), X_cmap.max(), Y_cmap.min(), Y_cmap.max()]
    
    # Smooth cmap data if option is specified
    if 'smoothData' in list(plotArgs.keys()) and plotArgs['smoothData'] is not {}:
        smoothDataArgDict =  plotArgs['smoothData']
        sigma = smoothDataArgDict['sigma']
        order = smoothDataArgDict['order']
        
        smoothData(Z_cmap, sigma, order, normType='max')
        
    # Set color map
    if 'cmap' in list(plotArgs.keys()):
        cmap = plotArgs['cmap']
    else:
        cmap = plt.cm.get_cmap('magma').reversed()
        
    # Set interpolation type for cmap
    if 'interpType' in list(plotArgs.keys()):
        interpType = plotArgs['interpType']
    else:
        interpType = 'None'
                
    #---------------#
    #-- Make plot --#
    #---------------#
    fig, ax = plt.subplots(figsize=(8,8))
    
    # Set the aspect so that resulting figure is a square
    aspect = (axisRange[1] - axisRange[0]) / (axisRange[3] - axisRange[2])
    
    # Make the cmap plot
    if 'cmap_range' in list(plotArgs.keys()) and plotArgs['cmap_range'] != ():
        vmin, vmax = plotArgs['cmap_range']
        im1 = ax.imshow(np.rot90(Z_cmap), cmap=cmap, vmin=vmin, vmax=vmax, aspect=aspect, extent=axisRange, interpolation=interpType) 
    else:
        im1 = ax.imshow(np.rot90(Z_cmap), cmap=cmap, aspect=aspect, extent=axisRange, interpolation=interpType)
        
    #---------------#
    #-- Set Style --#
    #---------------#  
    
    # Set plot limits
    ax.set_xlim([axisRange[0], axisRange[1]])
    ax.set_ylim([axisRange[2], axisRange[3]])

    # Plot colorbar and add lables  
    import matplotlib.ticker as ticker
    sfmt=ticker.ScalarFormatter(useMathText=True) 
    sfmt.set_powerlimits((0, 0))
    
    cb = plt.colorbar(im1, orientation='vertical', format=sfmt, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=axisTickSize)
    cb.set_label(zAxisTitle, size=axisLabelSize)
    cb.ax.yaxis.get_offset_text().set_fontsize(axisTickSize)
    cb.ax.yaxis.get_offset_text().set_horizontalalignment('center')
    
    # Set x and y axis labels
    ax.tick_params(axis='both', labelsize=axisTickSize)
    ax.set(xlabel=xAxisTitle, ylabel=yAxisTitle)
    ax.yaxis.label.set_size(axisLabelSize)
    ax.xaxis.label.set_size(axisLabelSize)
    ax.ticklabel_format(useMathText=True)   

    # Add plot title
    ax.set_title(plotTitle, fontsize=titleLabelSize)
    
    # Adjust axis ticks if desired
    if 'axisTicks' in list(plotArgs.keys()):
        subPlotArgDict = plotArgs['axisTicks']
        
        # Set them to specific values with option for minor ticks
        if subPlotArgDict['xticks'] != []:        
            ax.set_xticks(subPlotArgDict['xticks']) 
        if subPlotArgDict['yticks'] != []:        
            ax.set_yticks(subPlotArgDict['yticks'])
        if subPlotArgDict['minorticks']:
            ax.minorticks_on()
    
    #------------------#
    #-- Add Contours --#
    #------------------#
    if contourDict is not None:
        
        for key in list(contourDict.keys()):
            subContourDict = contourDict[key]
            
            X_surface    = subContourDict['X_surface']
            Y_surface    = subContourDict['Y_surface'] 
            Z_surface    = subContourDict['Z_surface'] 
            Z_constraint = subContourDict['Z_constraint']
            color        = [subContourDict['color']]
            linestyle    = [subContourDict['linestyle']]
            label        = subContourDict['label']
            fontsize     = subContourDict['fontsize']
            if 'inline' in list(subContourDict.keys()):
                inline = subContourDict['inline']
            else:
                inline = True
            
            CS = ax.contour(X_surface, Y_surface, Z_surface.T - Z_constraint, levels=[0], colors=color, linestyles=linestyle)

            # If option is selected, shade contour
            if 'shade' in list(subContourDict.keys()):
                
                shadeDict = subContourDict['shade']

                contour_coords = CS.collections[0].get_paths()[0].vertices
                x_coords  = contour_coords[:,0]
                y1_coords = contour_coords[:,1]
                y2_coords = shadeDict['yLimit']

                plt.fill_between(x_coords, y1_coords, y2_coords, color=shadeDict['color'], alpha=shadeDict['alpha'])
    
            CL = ax.clabel(CS, CS.levels, inline=inline, fmt=label, fontsize=fontsize)
            if inline==False:
                for l in CL:
                    delta_x   = subContourDict['delta_x']
                    delta_y   = subContourDict['delta_y']
                    delta_rot = subContourDict['delta_rot']
                    
                    l_x, l_y = list(l.get_position())
                    l_yNew = l_y + delta_y*l_y
                    l_xNew = l_x + delta_x*l_x
                    l.set_position((l_xNew,l_yNew))
                    
                    l_r = l.get_rotation()
                    l_rNew =  l_r + delta_rot
                    l.set_rotation(l_rNew)
    
    #--------------------#
    #-- Add Other Text --#
    #--------------------#
    if textDict is not None:
        for key in list(textDict.keys()):
            subTextDict = textDict[key]
            
            ax.text(subTextDict['x'], subTextDict['y'], subTextDict['text'], fontsize=subTextDict['fontsize'])
    
    #---------------# 
    #-- Save Plot --#
    #---------------# 
    if 'plotName' in list(plotArgs.keys()):
        plt.savefig(plotArgs['plotName'],  bbox_inches='tight', dpi=500)    
    
    plt.show()

    
def plotHeatMap(x, y, data, mask=None, xlabel=r'${\rm log}_{10}(m_{\rm DM}/{\rm GeV})$', ylabel=r'${\rm log}_{10}(f/{\rm GeV})$', zlabel=r'$\sigma_{\rm eff} {\rm[ GeV^{-2}]}$', logZ=True):

    plt.figure(figsize=(8, 8))
    
    if(logZ and data.min().min() > 0.):
        import math
        from matplotlib.colors import LogNorm
        log_norm = LogNorm(vmin=data.min().min(), vmax=data.max().max())
        
        cbar_ticks = [math.pow(10, i) for i in range(math.floor(math.log10(data.min().min())), 1+math.ceil(math.log10(data.max().max())))]

        heat_map = sns.heatmap(data, mask=mask, norm=log_norm, cbar_kws={"ticks": cbar_ticks, "label": zlabel})
    else:
        heat_map = sns.heatmap(data)
        
    # Set axis tick values
    xticks_labels = x
    plt.xticks(np.arange(data.shape[1]) + .5, labels=xticks_labels)

    yticks_labels = y
    plt.yticks(np.arange(data.shape[0]) + .5, labels=yticks_labels, rotation ='horizontal')


    # Set axis labels
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    heat_map.figure.axes[-1].yaxis.label.set_size(20) #https://newbedev.com/seaborn-heatmap-colorbar-label-font-size

    plt.show()
