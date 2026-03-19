CC = mpiicc
CFLAGS = -shared -fPIC -O3 -qopenmp -mkl
LIB = -lmpi -liomp5 -lpthread -lhdf5
LIB1 = -lmpi -liomp5 -lpthread
H5PATH = /public/apps/HDF5/1.12.1_intel
PYX = readh5.pyx CalFunc.pyx
SRC1 = readh5.c
SRC2 = CalFunc.c
SO1 = readh5.so
SO2 = CalFunc.so
OUT_MODE = 2>/dev/null
EXE = namd-epc

all:	c so exe

c:
	cython $(PYX)

so:
	@PYPATH=`which python | awk -F '/bin/python' '{print $$1}'`;\
	LIBPATH=$$PYPATH/lib;\
	H5INCPATH=$$PYPATH/include;\
	PYINCPATH=`ls $$PYPATH/include/*/Python.h | awk -F '/Python.h' '{print $$1}'`;\
	PYV=`echo $$PYINCPATH | awk -F '/' '{print $$NF}'`;\
	echo "python version: $$PYV";\
	NPYINCPATH=`ls $$PYPATH/lib/$$PYV/site-packages/numpy/core/include/numpy/arrayobject.h | awk -F '/numpy/arrayobject.h' '{print $$1}'`;\
	echo "Python include path: $$PYINCPATH";\
	echo "NumPy include path: $$NPYINCPATH";\
	$(CC) $(SRC1) $(CFLAGS) $(LIB) -L$(H5PATH)/lib -I$$PYINCPATH -I$$NPYINCPATH -I$(H5PATH)/include -o $(SO1) $(OUT_MODE);\
	$(CC) $(SRC2) $(CFLAGS) $(LIB1) -I$$PYINCPATH -I$$NPYINCPATH -o $(SO2) $(OUT_MODE)

exe:
	@PYEXE=$$(ls `which python`);\
	echo "#!$$PYEXE" > $(EXE);\
	echo "if __name__ == '__main__':" >> $(EXE);\
	echo "    from SurfHop import SurfHop" >> $(EXE);\
	echo "    SurfHop()" >> $(EXE);\
	chmod +x $(EXE);\
	echo "Generate $(EXE)"

clean: 
	rm -f $(SO1) $(SO2) $(EXE)
