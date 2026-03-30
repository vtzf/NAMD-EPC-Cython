CC = mpiicc
H5PATH = /public/apps/HDF5/1.12.1_intel

CFLAGS = -shared -fPIC -O3 -qopenmp -mkl
CFLAGS_HDF5 = -I$(H5PATH)/include -lhdf5 -L$(H5PATH)/lib

LIB = -lmpi -liomp5 -lpthread

OUT_MODE = 2>/dev/null
EXE = namd-epc

all: c so exe

c:
	cython readh5.pyx CalFunc.pyx

so:
	@PYPATH=`which python | awk -F '/bin/python' '{print $$1}'`;\
	PYINCPATH=`ls $$PYPATH/include/*/Python.h | awk -F '/Python.h' '{print $$1}'`;\
	PYV=`echo $$PYINCPATH | awk -F '/' '{print $$NF}'`;\
	NPYINCPATH=`ls $$PYPATH/lib/$$PYV/site-packages/numpy/core/include/numpy/arrayobject.h | awk -F '/numpy/arrayobject.h' '{print $$1}'`;\
	echo "python version: $$PYV";\
	echo "Python include path: $$PYINCPATH";\
	echo "NumPy include path: $$NPYINCPATH";\
	$(CC) readh5.c $(CFLAGS) $(CFLAGS_HDF5) $(LIB) -I$$PYINCPATH -I$$NPYINCPATH -o readh5.so $(OUT_MODE);\
	$(CC) CalFunc.c $(CFLAGS) $(LIB) -I$$PYINCPATH -I$$NPYINCPATH -o CalFunc.so $(OUT_MODE)

exe:
	@PYEXE=$$(ls `which python`);\
	echo "#!$$PYEXE" > $(EXE);\
	echo "if __name__ == '__main__':" >> $(EXE);\
	echo "    from SurfHop import SurfHop" >> $(EXE);\
	echo "    SurfHop()" >> $(EXE);\
	chmod +x $(EXE);\
	echo "Generate $(EXE)"

clean:
	rm -f *.so *.c $(EXE)
