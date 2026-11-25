CC = mpiicc
PYPATH = /public/apps/anaconda/anaconda3/2023.03
PYTHON := $(shell if [ -n "$(PYPATH)" ] && [ -x "$(PYPATH)/bin/python" ]; then printf '%s' "$(PYPATH)/bin/python"; else command -v python; fi)
H5PATH = /public/apps/HDF5/1.12.1_intel
SHM_MAX = 32212254720

CFLAGS = -shared -fPIC -O3 -qopenmp -mkl -DSHM_MAX_BYTES=$(SHM_MAX)
CFLAGS_HDF5 = -I$(H5PATH)/include -lhdf5 -L$(H5PATH)/lib

LIB = -lmpi -liomp5 -lpthread

OUT_MODE = 2>/dev/null
EXE = namd-epc

all: c so exe

c:
	$(PYTHON) -m cython readh5.pyx CalFunc.pyx

so:
	@PYINCPATH=`$(PYTHON) -c 'import sysconfig; print(sysconfig.get_paths()["include"])'`;\
	NPYINCPATH=`$(PYTHON) -c 'import numpy; print(numpy.get_include())'`;\
	echo "Python executable: $(PYTHON)";\
	echo "Python include path: $$PYINCPATH";\
	echo "NumPy include path: $$NPYINCPATH";\
	$(CC) readh5.c $(CFLAGS) $(CFLAGS_HDF5) $(LIB) -I$$PYINCPATH -I$$NPYINCPATH -o readh5.so $(OUT_MODE);\
	$(CC) CalFunc.c $(CFLAGS) $(LIB) -I$$PYINCPATH -I$$NPYINCPATH -o CalFunc.so $(OUT_MODE)

exe:
	@echo "#!$(PYTHON)" > $(EXE);\
	echo "if __name__ == '__main__':" >> $(EXE);\
	echo "    from SurfHop import SurfHop" >> $(EXE);\
	echo "    SurfHop()" >> $(EXE);\
	chmod +x $(EXE);\
	echo "Generate $(EXE)"

clean:
	rm -f *.so *.c $(EXE)
