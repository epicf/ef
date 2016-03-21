#include "vec3d.h"

Vec3d vec3d_init( double x, double y, double z )
{
    Vec3d v = { .x = {x,y,z} };
    return v;
}

Vec3d vec3d_zero()
{
    Vec3d v = { .x = {0.0, 0.0, 0.0} };
    return v;
}

double vec3d_x( Vec3d v )
{
    return v.x[0];
}

double vec3d_y( Vec3d v )
{
    return v.x[1];
}

double vec3d_z( Vec3d v )
{
    return v.x[2];
}

Vec3d vec3d_negate( Vec3d v )
{
    return vec3d_init( -vec3d_x(v), -vec3d_y(v), -vec3d_z(v) );
}

Vec3d vec3d_add( Vec3d v1, Vec3d v2 )
{
    return vec3d_init( vec3d_x(v1) + vec3d_x(v2), 
		       vec3d_y(v1) + vec3d_y(v2),
		       vec3d_z(v1) + vec3d_z(v2) );
}

Vec3d vec3d_sub( Vec3d v1, Vec3d v2 )
{
    return vec3d_init( vec3d_x(v1) - vec3d_x(v2), 
		       vec3d_y(v1) - vec3d_y(v2),
		       vec3d_z(v1) - vec3d_z(v2) );
}

double vec3d_dot_product( Vec3d v1, Vec3d v2 )
{
    
    return ( vec3d_x(v1) * vec3d_x(v2) + 
	     vec3d_y(v1) * vec3d_y(v2) +
	     vec3d_z(v1) * vec3d_z(v2) );
}

Vec3d vec3d_cross_product( Vec3d v1, Vec3d v2 )
{
    double prod_x = vec3d_y( v1 ) * vec3d_z( v2 ) - vec3d_z( v1 ) * vec3d_y( v2 );
    double prod_y = vec3d_z( v1 ) * vec3d_x( v2 ) - vec3d_x( v1 ) * vec3d_z( v2 );
    double prod_z = vec3d_x( v1 ) * vec3d_y( v2 ) - vec3d_y( v1 ) * vec3d_x( v2 );
    
    return vec3d_init( prod_x, prod_y, prod_z );
}

Vec3d vec3d_times_scalar( Vec3d v, double a )
{
    return vec3d_init( a * vec3d_x(v), 
		       a * vec3d_y(v),
		       a * vec3d_z(v) );    
}

void vec3d_print( Vec3d v )
{
    printf("(%e, %e, %e)",
	   vec3d_x(v), vec3d_y(v), vec3d_z(v) );
}

hid_t vec3d_hdf5_compound_type_for_memory()
{
    hid_t compound_type_for_mem;
    herr_t status;
    compound_type_for_mem = H5Tcreate( H5T_COMPOUND, sizeof(Vec3d) );
    vec3d_hdf5_status_check( compound_type_for_mem );
    status = H5Tinsert( compound_type_for_mem, "vec_x",
			HOFFSET( Vec3d, x ), H5T_NATIVE_DOUBLE );
    vec3d_hdf5_status_check( status );
    status = H5Tinsert( compound_type_for_mem, "vec_y",
			HOFFSET( Vec3d, x ) + sizeof(double), H5T_NATIVE_DOUBLE );
    vec3d_hdf5_status_check( status );
    status = H5Tinsert( compound_type_for_mem, "vec_z",
			HOFFSET( Vec3d, x ) + 2 * sizeof(double), H5T_NATIVE_DOUBLE );
    vec3d_hdf5_status_check( status );
    return compound_type_for_mem;
}

hid_t vec3d_hdf5_compound_type_for_file()
{
    hid_t compound_type_for_file;
    herr_t status;
    int double_size_in_file = 8;
    
    compound_type_for_file = H5Tcreate( H5T_COMPOUND, 3 * double_size_in_file );
    vec3d_hdf5_status_check( compound_type_for_file );
    status = H5Tinsert( compound_type_for_file, "vec_x", 0, H5T_IEEE_F64BE);
    vec3d_hdf5_status_check( status );
    status = H5Tinsert( compound_type_for_file, "vec_y", double_size_in_file, H5T_IEEE_F64BE );
    vec3d_hdf5_status_check( status );
    status = H5Tinsert( compound_type_for_file, "vec_z", 2 * double_size_in_file, H5T_IEEE_F64BE );
    vec3d_hdf5_status_check( status );

    return compound_type_for_file;
}

void vec3d_hdf5_status_check( herr_t status )
{
    if( status < 0 ){
	printf( "Something went wrong while creating compound datatypes for Vec3d. Aborting.\n" );
	exit( EXIT_FAILURE );
    }
}
