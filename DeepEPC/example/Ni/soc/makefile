CC = mpiicc
H5PATH = /public/apps/HDF5/1.12.1_intel

CFLAGS = -shared -fPIC -O3 -qopenmp 
CFLAGS_MKL = -DMKL_LP64 -mkl=parallel -I${MKLROOT}/include
CFLAGS_HDF5 = -I${H5PATH}/include -lhdf5 -L${H5PATH}/lib

LIB = -lmpi -liomp5 -lpthread -lm -ldl
LIB_MKL = -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core
LIB_MKL_CBLAS = ${CFLAGS_MKL} ${LIB_MKL}
LIB_MKL_BLACS = ${LIB_MKL_CBLAS} -lmkl_scalapack_lp64 -lmkl_blacs_intelmpi_lp64

PYX = readdrSH.pyx epcfunc_nl.pyx readhamilsparse.pyx epcfuncsparse.pyx \
      readdrSH_nc.pyx epcfunc_nl_nc.pyx readhamilsparse_nc.pyx epcfuncsparse_nc.pyx

SRC = readdrSH.c
SRC1 = epcfunc_nl.c
SRC2 = readhamilsparse.c
SRC3 = epcfuncsparse.c
SRC4 = readdrSH_nc.c
SRC5 = epcfunc_nl_nc.c
SRC6 = readhamilsparse_nc.c
SRC7 = epcfuncsparse_nc.c

SO = readdrSH.so
SO1 = epcfunc_nl.so
SO2 = readhamilsparse.so
SO3 = epcfuncsparse.so
SO4 = readdrSH_nc.so
SO5 = epcfunc_nl_nc.so
SO6 = readhamilsparse_nc.so
SO7 = epcfuncsparse_nc.so

OUT_MODE = 2>/dev/null

all:	c so

c:
	cython $(PYX)

so:
	@PYPATH=`which python | awk -F '/bin/python' '{print $$1}'`;\
	LIBPATH=$$PYPATH/lib;\
	PYINCPATH=`ls $$PYPATH/include/*/Python.h | awk -F '/Python.h' '{print $$1}'`;\
	PYV=`echo $$PYINCPATH | awk -F '/' '{print $$NF}'`;\
	echo "python version: $$PYV";\
	echo "Python path: $$PYPATH";\
	echo "Python include path: $$PYINCPATH";\
	$(CC) $(SRC) $(CFLAGS) $(CFLAGS_HDF5) $(LIB) $(LIB_MKL_BLACS) -I$$PYINCPATH -o $(SO) $(OUT_MODE);\
	$(CC) $(SRC1) $(CFLAGS) $(LIB) $(LIB_MKL_CBLAS) -I$$PYINCPATH -o $(SO1) $(OUT_MODE);\
	$(CC) $(SRC2) $(CFLAGS) $(CFLAGS_HDF5) $(LIB) -I$$PYINCPATH -o $(SO2) $(OUT_MODE);\
	$(CC) $(SRC3) $(CFLAGS) $(LIB) $(LIB_MKL_CBLAS) -I$$PYINCPATH -o $(SO3) $(OUT_MODE);\
	$(CC) $(SRC4) $(CFLAGS) $(CFLAGS_HDF5) $(LIB) $(LIB_MKL_BLACS) -I$$PYINCPATH -o $(SO4) $(OUT_MODE);\
	$(CC) $(SRC5) $(CFLAGS) $(LIB) $(LIB_MKL_CBLAS) -I$$PYINCPATH -o $(SO5) $(OUT_MODE);\
	$(CC) $(SRC6) $(CFLAGS) $(CFLAGS_HDF5) $(LIB) -I$$PYINCPATH -o $(SO6) $(OUT_MODE);\
	$(CC) $(SRC7) $(CFLAGS) $(LIB) $(LIB_MKL_CBLAS) -I$$PYINCPATH -o $(SO7) $(OUT_MODE)

clean: 
	rm -f *.so
