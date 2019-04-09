C++ minimal multi thread executable reading fits files
from validation_data_hsc by `lsst::afw::table::SourceCatalog::readFits`
in 2 threads. As of DM-18695, DM-19212, this file crashes, see
example output.

Build notes
-----------

* This program is a dummy standalone package.

* Build and setup lsstsw environment (up to afw package).
 
* Build `ptest` by ``scons``.
 
* Make symlink `validation_data_hsc` in current directory
  pointing to the validation_data_hsc package directory.

* Run ``src/ptest``.


* sconsUtils reads the `buildOpts.py` file for command line options
   at the package top level only.
   
* `eupspkg build` however picks up environment variables and passes
  on to `scons` as cmd-line arguments. If no variable is set,
  defaults are passed on. Therefore  `opt=g`  in `buildOpts.py` is
  always disrespected if using `rebuild` for lsstsw rebuilding.
  
* Use ``EUPSPKG_SCONSFLAGS="opt=g archflags=-pthread"`` for
  `rebuild`. Note, as of DM-18695, only scons configured C++ builds
  are affected by this.

* To build a complete lsstsw stack to debug thread safety by this example
  from scratch::

     export EUPSPKG_SCONSFLAGS="opt=g archflags=-pthread"
     rebuild -r tickets/DM-18695 afw
     setup -t bNNNN afw

  Then in your own `afw` clone::

     # -pthread is defined in SConscript for ptest in examples/threading
     scons opt=g examples/threading
     ln -s path_validation_data_hsc validation_data_hsc
     examples/threading/ptest
     
