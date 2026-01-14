CC = mpiicc
H5PATH = /home/huangdq_group/tuyouyou/data/opt/hdf5/

CFLAGS = -shared -fPIC -O3 -qopenmp 
CFLAGS_MKL = -DMKL_LP64 -mkl=parallel -I${MKLROOT}/include
CFLAGS_HDF5 = -I${H5PATH}/include -lhdf5 -L${H5PATH}/lib

LIB = -lmpi -liomp5 -lpthread -lm -ldl
LIB_MKL = -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core
LIB_MKL_CBLAS = ${CFLAGS_MKL} ${LIB_MKL}
LIB_MKL_BLACS = ${LIB_MKL_CBLAS} -lmkl_scalapack_lp64 -lmkl_blacs_intelmpi_lp64

PYX = readdrSH.pyx epcfunc_nl.pyx readhamilsparse.pyx epcfuncsparse.pyx

SRC = readdrSH.c
SRC1 = epcfunc_nl.c
SRC2 = readhamilsparse.c
SRC3 = epcfuncsparse.c

SO = readdrSH.so
SO1 = epcfunc_nl.so
SO2 = readhamilsparse.so
SO3 = epcfuncsparse.so

OUT_MODE = 2>/dev/null

all:	c so exe

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
	$(CC) $(SRC3) $(CFLAGS) $(LIB) $(LIB_MKL_CBLAS) -I$$PYINCPATH -o $(SO3) $(OUT_MODE)

exe:
	@PYEXE=$$(ls `which python`);\
	echo "#!$$PYEXE" > $(EXE);\
	echo "if __name__ == '__main__':" >> $(EXE);\
	echo "    from SurfHop import SurfHop" >> $(EXE);\
	echo "    SurfHop()" >> $(EXE);\
	chmod +x $(EXE);\
	echo "Generate $(EXE)"

clean: 
	rm -f $(SO) $(SO1) $(SO2) $(SO3) $(EXE)
