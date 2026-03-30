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

SRC_BLACS = readdrSH.c readdrSH_nc.c
SRC_HDF5  = readhamilsparse.c readhamilsparse_nc.c
SRC_CBLAS = epcfunc_nl.c epcfunc_nl_nc.c epcfuncsparse.c epcfuncsparse_nc.c

OUT_MODE = 2>/dev/null

all: c so

c:
	cython $(PYX)

so:
	@PYPATH=`which python | awk -F '/bin/python' '{print $$1}'`;\
	PYINCPATH=`ls $$PYPATH/include/*/Python.h | awk -F '/Python.h' '{print $$1}'`;\
	PYV=`echo $$PYINCPATH | awk -F '/' '{print $$NF}'`;\
	echo "python version: $$PYV";\
	echo "Python path: $$PYPATH";\
	echo "Python include path: $$PYINCPATH";\
	for src in $(SRC_BLACS); do \
		$(CC) $$src $(CFLAGS) $(CFLAGS_HDF5) $(LIB) $(LIB_MKL_BLACS) -I$$PYINCPATH -o $${src%.c}.so $(OUT_MODE); \
	done;\
	for src in $(SRC_HDF5); do \
		$(CC) $$src $(CFLAGS) $(CFLAGS_HDF5) $(LIB) -I$$PYINCPATH -o $${src%.c}.so $(OUT_MODE); \
	done;\
	for src in $(SRC_CBLAS); do \
		$(CC) $$src $(CFLAGS) $(LIB) $(LIB_MKL_CBLAS) -I$$PYINCPATH -o $${src%.c}.so $(OUT_MODE); \
	done

clean:
	rm -f *.so *.c
