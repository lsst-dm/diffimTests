import sys

try:
    import lsstDebug

    print "Importing debug settings..."
    def DebugInfo(name):
        di = lsstDebug.getInfo(name)
        #di.display = True # N.b. lsstDebug.Info(name) would call us recursively
        if name == "lsst.pipe.tasks.matchBackgrounds":
            di.display = False
            di.savefig = True
            di.savefits = False
            di.figpath = "/lsst/home/yusra/figures/"
        if name == "lsst.obs.sdss.selectFluxMag0":
            di.display = True
        if name == "lsst.ip.diffim.psfMatch":
            di.display = True                 # global; _solve function
            di.maskTransparency = 80          # ds9 mask transparency
            di.displayCandidates = True       # show all the candidates and residuals
            di.displayKernelBasis = True      # show kernel basis functions
            di.displayKernelMosaic = True     # show kernel realized across the image
            di.plotKernelSpatialModel = False # show coefficients of spatial model
            di.showBadCandidates = True       # show the bad candidates (red) along with good (green)
        elif name == "lsst.ip.diffim.imagePsfMatch":
            di.display = False                # global
            di.maskTransparency = 80          # ds9 mask transparency
            di.displayTemplate = True         # show full (remapped) template
            di.displaySciIm = True            # show science image to match to
            di.displaySpatialCells = True     # show spatial cells
            di.displayDiffIm = True           # show difference image
            di.showBadCandidates = True       # show the bad candidates (red) along with good (green)
        elif name == "lsst.pipe.tasks.imageDifference":
            di.display =True                # global
            di.maskTransparency = 50          # ds9 mask transparency
            di.showPixelResiduals = True      # histograms of diffim / sqrt(variance)
            di.showDiaSources = True          # display diffim-detected sources
            
        return di
    lsstDebug.Info = DebugInfo
    lsstDebug.frame = 1
 
except ImportError:
    print >> sys.stderr, "Unable to import lsstDebug;  not setting display intelligently"
