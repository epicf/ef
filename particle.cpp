#include "particle.h"

Particle::Particle( int id, double charge, double mass, Vec3d position, Vec3d momentum ) :
    id( id ),
    charge( charge ),
    mass( mass ),
    position( position ),
    momentum( momentum ),
    momentum_is_half_time_step_shifted( false )
{ }


void Particle::update_position( double dt )
{
    Vec3d pos_shift;            
    pos_shift = vec3d_times_scalar( momentum, dt / mass );
    position = vec3d_add( position, pos_shift );
}

void Particle::print()
{
    std::cout.setf( std::ios::scientific );
    std::cout.precision( 3 );    
    std::cout << "Particle: ";
    std::cout << "id: " << id << ", ";
    std::cout << "charge = " << charge << " mass = " << mass << ", ";
    std::cout << "pos(x,y,z) = ("
	      << vec3d_x( position ) << ", "
	      << vec3d_y( position ) << ", "
	      << vec3d_z( position ) << "), ";
    std::cout << "momentum(px,py,pz) = ("
	      << vec3d_x( momentum ) << ", "
	      << vec3d_y( momentum ) << ", "
	      << vec3d_z( momentum ) << ")";
    std::cout << std::endl;
    return;
}


void Particle::print_short()
{    
    std::cout.setf( std::ios::scientific );
    std::cout.precision( 2 );    
    std::cout << "id: " << id << " "
	      << "x = " << vec3d_x( position ) << " "
	      << "y = " << vec3d_y( position ) << " "
	      << "z = " << vec3d_z( position ) << " "	
	      << "px = " << vec3d_x( momentum ) << " "
	      << "py = " <<  vec3d_y( momentum ) << " "
	      << "pz = " <<  vec3d_z( momentum ) << " "
	      << std::endl;
    return;
}


hid_t HDF5_buffer_for_Particle_compound_type_for_memory()
{
    hid_t compound_type_for_mem;
    herr_t status;

    hid_t vec3d_compound_type_for_mem;
    vec3d_compound_type_for_mem = vec3d_hdf5_compound_type_for_memory();
    
    compound_type_for_mem = H5Tcreate( H5T_COMPOUND, sizeof(HDF5_buffer_for_Particle) );
        HDF5_buffer_for_Particle_hdf5_status_check( compound_type_for_mem );

    status = H5Tinsert( compound_type_for_mem, "id",
			HOFFSET( HDF5_buffer_for_Particle, id ), H5T_NATIVE_INT );
        HDF5_buffer_for_Particle_hdf5_status_check( status );
    status = H5Tinsert( compound_type_for_mem, "position",
			HOFFSET( HDF5_buffer_for_Particle, position ), vec3d_compound_type_for_mem );
        HDF5_buffer_for_Particle_hdf5_status_check( status );
    status = H5Tinsert( compound_type_for_mem, "momentum",
			HOFFSET( HDF5_buffer_for_Particle, momentum ), vec3d_compound_type_for_mem );
        HDF5_buffer_for_Particle_hdf5_status_check( status );
    status = H5Tinsert( compound_type_for_mem, "mpi_proc_rank",
			HOFFSET( HDF5_buffer_for_Particle, mpi_proc_rank ), H5T_NATIVE_INT );
        HDF5_buffer_for_Particle_hdf5_status_check( status );

    status = H5Tclose( vec3d_compound_type_for_mem );
        HDF5_buffer_for_Particle_hdf5_status_check( status );
    
    return compound_type_for_mem;
}

hid_t HDF5_buffer_for_Particle_compound_type_for_file()
{
    hid_t compound_type_for_file;
    herr_t status;

    hid_t vec3d_compound_type_for_file;
    vec3d_compound_type_for_file = vec3d_hdf5_compound_type_for_file();
    
    compound_type_for_file = H5Tcreate( H5T_COMPOUND, 4 + 3*8 + 3*8 + 4 );
        HDF5_buffer_for_Particle_hdf5_status_check( compound_type_for_file );

    status = H5Tinsert( compound_type_for_file, "id", 0, H5T_STD_I32BE );
        HDF5_buffer_for_Particle_hdf5_status_check( status );
    status = H5Tinsert( compound_type_for_file, "position", 4, vec3d_compound_type_for_file );
        HDF5_buffer_for_Particle_hdf5_status_check( status );
    status = H5Tinsert( compound_type_for_file, "momentum", 4 + 3*8, vec3d_compound_type_for_file );
        HDF5_buffer_for_Particle_hdf5_status_check( status );
    status = H5Tinsert( compound_type_for_file, "mpi_proc_rank", 4 + 3*8 + 3*8, H5T_STD_I32BE );
        HDF5_buffer_for_Particle_hdf5_status_check( status );

    status = H5Tclose( vec3d_compound_type_for_file );
        HDF5_buffer_for_Particle_hdf5_status_check( status );
    
    return compound_type_for_file;
}

void HDF5_buffer_for_Particle_hdf5_status_check( herr_t status )
{
    if( status < 0 ){
	std::cout << "Something went wrong while creating compound datatypes for HDF5_buffer_for_Particle. Aborting."
		  << std::endl;
	exit( EXIT_FAILURE );
    }
}
