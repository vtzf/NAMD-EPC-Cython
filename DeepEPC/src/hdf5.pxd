from libc.time cimport time_t

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
    cdef herr_t H5Fclose(hid_t file_id)
    cdef hid_t H5Dget_type(hid_t dset_id)

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

    # HDF5 >= 1.10.5
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
