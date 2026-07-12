from libc.time cimport time_t

# ---------------------------------------------------------------------------
# Compatibility shim for H5Ovisit API across HDF5 versions
#
# HDF5 < 1.10.3 (e.g. 1.8.x):
#   - H5Ovisit(obj_id, idx_type, order, H5O_iterate_t op, op_data)
#   - callback: typedef herr_t (*H5O_iterate_t)(hid_t, const char*, const H5O_info_t*, void*)
#
# HDF5 >= 1.10.3:
#   - H5Ovisit1(obj_id, idx_type, order, H5O_iterate1_t op, op_data)
#   - callback: typedef herr_t (*H5O_iterate1_t)(hid_t, const char*, const H5O_info1_t*, void*)
#   - H5O_info1_t is structurally identical to the old H5O_info_t
#
# HDF5 >= 1.12.0:
#   - H5Ovisit1 is deprecated (H5Ovisit3 is the new default) but still available
#
# The shim below aliases H5Ovisit1 / H5O_info1_t / H5O_iterate1_t to the
# old unversioned names when compiling against HDF5 < 1.10.3, so the rest
# of the Cython code can always use the versioned names.
# ---------------------------------------------------------------------------
cdef extern from *:
    """
    #include "hdf5.h"
    #if !H5_VERSION_GE(1,10,3)
    /* HDF5 < 1.10.3: versioned names do not exist; create aliases. */
    typedef H5O_info_t    H5O_info1_t;
    typedef H5O_iterate_t H5O_iterate1_t;
    static herr_t H5Ovisit1(hid_t obj_id, H5_index_t idx_type,
                             H5_iter_order_t order,
                             H5O_iterate1_t op, void *op_data) {
        return H5Ovisit(obj_id, idx_type, order, op, op_data);
    }
    #endif
    """
    pass

cdef extern from "hdf5.h":
    ctypedef long hid_t
    ctypedef int herr_t
    cdef int H5T_NATIVE_INT
    cdef int H5T_NATIVE_DOUBLE
    cdef hid_t H5S_ALL
    cdef unsigned int H5F_ACC_RDONLY
    cdef unsigned int H5P_DEFAULT
    cdef hid_t H5Fopen(
        char* filename, unsigned int flags, hid_t access_plist
    )
    cdef hid_t H5Dopen(
        hid_t file_id, const char* name, hid_t dapl_id
    )
    cdef herr_t H5Dread(
        hid_t dset_id, hid_t mem_type_id, hid_t mem_space_id,
        hid_t file_space_id, hid_t plist_id, void* buf
    )
    cdef herr_t H5Dclose(hid_t dset_id)
    cdef herr_t H5Tclose(hid_t type_id)
    cdef herr_t H5Fclose(hid_t file_id)
    cdef hid_t H5Dget_type(hid_t dset_id)
    cdef int H5T_COMPOUND
    cdef hid_t H5Tcreate(int cls, size_t size)
    cdef herr_t H5Tinsert(hid_t parent_id, char* name, size_t offset, hid_t member_id)

    ctypedef signed long long haddr_t
    ctypedef long long hsize_t
    ctypedef struct space:
        hsize_t total
        hsize_t meta
        hsize_t mesg
        hsize_t free
    ctypedef struct mesg:
        unsigned long present
        unsigned long shared
    ctypedef struct H5O_hdr_info_t:
        unsigned version
        unsigned nmesgs
        unsigned nchunks
        unsigned flags
        space space
        mesg mesg
    ctypedef struct H5_ih_info_t:
        hsize_t     index_size,
        hsize_t     heap_size
    cdef struct meta_size:
        H5_ih_info_t   obj,
        H5_ih_info_t   attr
    ctypedef int htri_t
    cdef htri_t H5Lexists(hid_t loc_id, char* name, hid_t lapl_id)
    ctypedef enum H5_index_t:
        H5_INDEX_UNKNOWN = -1,
        H5_INDEX_NAME,
        H5_INDEX_CRT_ORDER,
        H5_INDEX_N
    ctypedef enum H5_iter_order_t:
        H5_ITER_UNKNOWN = -1,
        H5_ITER_INC,
        H5_ITER_DEC,
        H5_ITER_NATIVE,
        H5_ITER_N
    ctypedef enum H5O_type_t:
        H5O_TYPE_UNKNOWN = -1,
        H5O_TYPE_GROUP,
        H5O_TYPE_DATASET,
        H5O_TYPE_NAMED_DATATYPE,
        H5O_TYPE_NTYPES

    # H5O_info1_t / H5O_iterate1_t / H5Ovisit1 are available natively on
    # HDF5 >= 1.10.3, and provided via the compatibility shim above on
    # HDF5 < 1.10.3 (where the old unversioned H5O_info_t layout is identical).
    ctypedef struct H5O_info1_t:
        unsigned long   fileno
        haddr_t         addr
        H5O_type_t      type
        unsigned        rc
        time_t          atime
        time_t          mtime
        time_t          ctime
        time_t          btime
        hsize_t         num_attrs
        H5O_hdr_info_t  hdr
        meta_size       meta_size
    ctypedef herr_t (*H5O_iterate1_t)(
        hid_t obj, char* name, H5O_info1_t* info, void* op_data
    ) except *
    cdef herr_t H5Ovisit1(
        hid_t obj_id, H5_index_t idx_type,
        H5_iter_order_t order, H5O_iterate1_t op, void* op_data
    )
