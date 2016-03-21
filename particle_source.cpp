#include "particle_source.h"

void check_and_warn_if_not( const bool &should_be, const std::string &message );

Particle_source::Particle_source( 
    Config &conf, 
    Source_config_part &src_conf )
{
    check_correctness_of_related_config_fields( conf, src_conf );
    set_parameters_from_config( src_conf );
    generate_initial_particles();
}

void Particle_source::check_correctness_of_related_config_fields( 
    Config &conf, 
    Source_config_part &src_conf )
{
    particle_source_initial_number_of_particles_gt_zero( conf, src_conf );
    particle_source_particles_to_generate_each_step_ge_zero( conf, src_conf );
    particle_source_x_left_ge_zero( conf, src_conf );
    particle_source_x_left_le_particle_source_x_right( conf, src_conf );
    particle_source_x_right_le_grid_x_size( conf, src_conf );
    particle_source_y_bottom_ge_zero( conf, src_conf );
    particle_source_y_bottom_le_particle_source_y_top( conf, src_conf );
    particle_source_y_top_le_grid_y_size( conf, src_conf );
    particle_source_z_near_ge_zero( conf, src_conf );
    particle_source_z_near_le_particle_source_z_far( conf, src_conf );
    particle_source_z_far_le_grid_z_size( conf, src_conf );
    particle_source_temperature_gt_zero( conf, src_conf );
    particle_source_mass_gt_zero( conf, src_conf );
}

void Particle_source::set_parameters_from_config( Source_config_part &src_conf )
{
    name = src_conf.particle_source_name;
    initial_number_of_particles = src_conf.particle_source_initial_number_of_particles;
    particles_to_generate_each_step = 
	src_conf.particle_source_particles_to_generate_each_step;
    xleft = src_conf.particle_source_x_left;
    xright = src_conf.particle_source_x_right;
    ytop = src_conf.particle_source_y_top;
    ybottom = src_conf.particle_source_y_bottom;
    znear = src_conf.particle_source_z_near;
    zfar = src_conf.particle_source_z_far;
    mean_momentum = vec3d_init( src_conf.particle_source_mean_momentum_x, 
				src_conf.particle_source_mean_momentum_y,
				src_conf.particle_source_mean_momentum_z );
    temperature = src_conf.particle_source_temperature;
    charge = src_conf.particle_source_charge;
    mass = src_conf.particle_source_mass;    
    // Random number generator
    // Simple approach: use different seed for each proccess.
    // Other way would be to synchronize the state of the rnd_gen
    //    between each processes after each call to it.    
    int mpi_process_rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &mpi_process_rank );    
    unsigned seed = 0 + 1000000 * mpi_process_rank;
    rnd_gen = std::default_random_engine( seed );
    // Initial id
    max_id = 0;
}

void Particle_source::generate_initial_particles()
{
    //particles.reserve( initial_number_of_particles );
    generate_num_of_particles( initial_number_of_particles );
}

void Particle_source::generate_each_step()
{
    //particles.reserve( particles.size() + particles_to_generate_each_step );
    generate_num_of_particles( particles_to_generate_each_step );
}
    
void Particle_source::generate_num_of_particles( int num_of_particles )
{
    Vec3d pos, mom;
    std::vector<int> vec_of_ids;
    int num_of_particles_for_this_proc;

    int mpi_n_of_proc, mpi_process_rank;
    MPI_Comm_size( MPI_COMM_WORLD, &mpi_n_of_proc );
    MPI_Comm_rank( MPI_COMM_WORLD, &mpi_process_rank );    

    num_of_particles_for_this_proc = num_of_particles_for_each_process( num_of_particles );
    populate_vec_of_ids( vec_of_ids, num_of_particles_for_this_proc ); 
    for ( int i = 0; i < num_of_particles_for_this_proc; i++ ) {
	pos = uniform_position_in_cube( xleft, ytop, znear,
					xright, ybottom, zfar,
					rnd_gen );
	mom = maxwell_momentum_distr( mean_momentum, temperature, mass, rnd_gen );
	particles.emplace_back( vec_of_ids[i], charge, mass, pos, mom );
    }
}

int Particle_source::num_of_particles_for_each_process( int total_num_of_particles )
{
    int rest;
    int mpi_n_of_proc, mpi_process_rank;
    MPI_Comm_size( MPI_COMM_WORLD, &mpi_n_of_proc );
    MPI_Comm_rank( MPI_COMM_WORLD, &mpi_process_rank );    
    
    int num_of_particles_for_this_proc = total_num_of_particles / mpi_n_of_proc;
    rest = total_num_of_particles % mpi_n_of_proc;
    if( mpi_process_rank < rest ){
	num_of_particles_for_this_proc++;
	// Processes with lesser ranks will accumulate
	// more particles.
	// This seems unessential.
    }
    return num_of_particles_for_this_proc;
}

void Particle_source::populate_vec_of_ids(
    std::vector<int> &vec_of_ids, int num_of_particles_for_this_proc )
{
    int mpi_n_of_proc, mpi_process_rank;
    MPI_Comm_size( MPI_COMM_WORLD, &mpi_n_of_proc );
    MPI_Comm_rank( MPI_COMM_WORLD, &mpi_process_rank );    

    vec_of_ids.reserve( num_of_particles_for_this_proc );
    
    for( int proc = 0; proc < mpi_n_of_proc; proc++ ){
	if( mpi_process_rank == proc ){
	    for( int i = 0; i < num_of_particles_for_this_proc; i++ ){
		vec_of_ids.push_back( max_id++ );
	    }	    
	}
	MPI_Bcast( &max_id, 1, MPI_INT, proc, MPI_COMM_WORLD );
    }
}

// int Particle_source::generate_particle_id( const int number, const int proc )
// {
//     max_id++;
//     MPI_Bcast( &max_id, 1, MPI_UNSIGNED, proc, MPI_COMM_WORLD );
//     return max_id;     
// }

Vec3d Particle_source::uniform_position_in_cube( 
    const double xleft,  const double ytop, const double znear,
    const double xright, const double ybottom, const double zfar,
    std::default_random_engine &rnd_gen )
{
    return vec3d_init( random_in_range( xleft, xright, rnd_gen ), 
		       random_in_range( ybottom, ytop, rnd_gen ),
		       random_in_range( znear, zfar, rnd_gen ) );
}

double Particle_source::random_in_range( 
    const double low, const double up, 
    std::default_random_engine &rnd_gen )
{
    std::uniform_real_distribution<double> uniform_distr( low, up );
    return uniform_distr( rnd_gen );
}

Vec3d Particle_source::maxwell_momentum_distr(
    const Vec3d mean_momentum, const double temperature, const double mass, 
    std::default_random_engine &rnd_gen )
{    
    double maxwell_gauss_std_mean_x = vec3d_x( mean_momentum );
    double maxwell_gauss_std_mean_y = vec3d_y( mean_momentum );
    double maxwell_gauss_std_mean_z = vec3d_z( mean_momentum );
    double maxwell_gauss_std_dev = sqrt( mass * temperature );
    std::normal_distribution<double> 
	normal_distr_x( maxwell_gauss_std_mean_x, maxwell_gauss_std_dev );
    std::normal_distribution<double> 
	normal_distr_y( maxwell_gauss_std_mean_y, maxwell_gauss_std_dev );
    std::normal_distribution<double> 
	normal_distr_z( maxwell_gauss_std_mean_z, maxwell_gauss_std_dev );

    Vec3d mom;
    mom = vec3d_init( normal_distr_x( rnd_gen ),
		      normal_distr_y( rnd_gen ),
		      normal_distr_z( rnd_gen ) );		     
    mom = vec3d_times_scalar( mom, 1.0 ); // recheck
    return mom;
}

void Particle_source::update_particles_position( double dt )
{
    for ( auto &p : particles )
	p.update_position( dt );
}


void Particle_source::print_particles()
{
    std::cout << "Source name: " << name << std::endl;
    for ( auto& p : particles  ) {	
	p.print_short();
    }
    return;
}

void Particle_source::write_to_file( hid_t group_id )
{
    // todo: print total N of particles
    // int mpi_process_rank;
    // MPI_Comm_rank( MPI_COMM_WORLD, &mpi_process_rank );    
    // if( mpi_process_rank == 0 ){

    std::cout << "Source name = " << name << ", "
	      << "number of particles = " << particles.size()
	      << std::endl;
    
    std::string table_of_particles_name = name;

    write_hdf5_particles( group_id, table_of_particles_name );

    if( total_particles_across_all_processes() != 0 ){
	// todo: attempt to create an attribute for the empty dataset results in error
	write_hdf5_source_parameters( group_id, table_of_particles_name );
    } else {
	std::cout << "Warning: Number of particles of " << name << " source is zero." << std::endl;
	std::cout << "Warning: Skipping attributes for " << name << std::endl;
	std::cout << "Known bug: attemp to write an attribute for an empty dataset would result in error." << std::endl;
    }
    
    return;
}

void Particle_source::write_hdf5_particles( hid_t group_id, std::string table_of_particles_name )
{
    // todo: remove
    int mpi_n_of_proc, mpi_process_rank;
    MPI_Comm_size( MPI_COMM_WORLD, &mpi_n_of_proc );
    MPI_Comm_rank( MPI_COMM_WORLD, &mpi_process_rank );    
    //
    herr_t status;
    hid_t filespace, memspace, dset;
    hid_t compound_type_for_mem, compound_type_for_file; 
    hid_t plist_id;
    int rank = 1;
    hsize_t dims[rank], subset_dims[rank], subset_offset[rank];
    dims[0] = total_particles_across_all_processes();
    
    // todo: dst_buf should be removed.
    // currently it is used to avoid any problems of
    // working with Particles class, which is a C++ class
    // and not a plain C datastructure
    HDF5_buffer_for_Particle *dst_buf = new HDF5_buffer_for_Particle[ particles.size() ];
    for( unsigned int i = 0; i < particles.size(); i++ ){
	dst_buf[i].id = particles[i].id;
	dst_buf[i].position = particles[i].position;
	dst_buf[i].momentum = particles[i].momentum;
	dst_buf[i].mpi_proc_rank = mpi_process_rank;
    }     

    compound_type_for_mem = HDF5_buffer_for_Particle_compound_type_for_memory();
    compound_type_for_file = HDF5_buffer_for_Particle_compound_type_for_file();
    plist_id = H5Pcreate( H5P_DATASET_XFER ); hdf5_status_check( plist_id );
    status = H5Pset_dxpl_mpio( plist_id, H5FD_MPIO_COLLECTIVE ); hdf5_status_check( status );

    subset_dims[0] = particles.size();
    subset_offset[0] = data_offset_for_each_process_for_1d_dataset();

    // todo: remove
    // std::cout << "particles "
    // 	      << "total = " << particles.size() << " "
    // 	      << "proc_n = " << mpi_process_rank << " "
    // 	      << "count = " << subset_dims[0] << " "
    // 	      << "offset = " << subset_offset[0] << std::endl;
    //
    
    memspace = H5Screate_simple( rank, subset_dims, NULL ); hdf5_status_check( memspace );
    filespace = H5Screate_simple( rank, dims, NULL ); hdf5_status_check( filespace );
    status = H5Sselect_hyperslab( filespace, H5S_SELECT_SET, subset_offset, NULL, subset_dims, NULL );
    hdf5_status_check( status );
    
    dset = H5Dcreate( group_id, ("./" + table_of_particles_name).c_str(),
    		      compound_type_for_file, filespace,
    		      H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT ); hdf5_status_check( dset );
    status = H5Dwrite( dset, compound_type_for_mem,
    		       memspace, filespace, plist_id, dst_buf ); hdf5_status_check( status );
    status = H5Dclose( dset ); hdf5_status_check( status );

    status = H5Sclose( filespace ); hdf5_status_check( status );
    status = H5Sclose( memspace ); hdf5_status_check( status );
    status = H5Pclose( plist_id ); hdf5_status_check( status );
    status = H5Tclose( compound_type_for_file ); hdf5_status_check( status );
    status = H5Tclose( compound_type_for_mem );	hdf5_status_check( status );    
    delete[] dst_buf;
}

int Particle_source::total_particles_across_all_processes()
{
    int n_of_particles = particles.size();
    int total_n_of_particles;
    int single_element = 1;

    MPI_Allreduce( &n_of_particles, &total_n_of_particles, single_element,
		   MPI_INT, MPI_SUM, MPI_COMM_WORLD );

    return total_n_of_particles;
}

int Particle_source::data_offset_for_each_process_for_1d_dataset()
{    
    int mpi_n_of_proc, mpi_process_rank;
    MPI_Comm_size( MPI_COMM_WORLD, &mpi_n_of_proc );
    MPI_Comm_rank( MPI_COMM_WORLD, &mpi_process_rank );    

    int offset = 0;
    int n_of_particles = particles.size();
    int single_element = 1;
    int *n_of_particles_at_each_proc = new int[ mpi_n_of_proc ];

    MPI_Allgather( &n_of_particles, single_element, MPI_INT,
		   n_of_particles_at_each_proc, single_element, MPI_INT,
		   MPI_COMM_WORLD );
    
    for( int i = 1; i <= mpi_process_rank; i++ ){
	offset = offset + n_of_particles_at_each_proc[i - 1];
    }

    delete[] n_of_particles_at_each_proc;
    
    return offset;
}


void Particle_source::write_hdf5_source_parameters( hid_t group_id,
						    std::string table_of_particles_name )
{
    herr_t status;
    int single_element = 1;
    double mean_mom_x = vec3d_x( mean_momentum );
    double mean_mom_y = vec3d_y( mean_momentum );
    double mean_mom_z = vec3d_z( mean_momentum );

    status = H5LTset_attribute_double( group_id, table_of_particles_name.c_str(),
    				       "xleft", &xleft, single_element );
    hdf5_status_check( status );
    status = H5LTset_attribute_double( group_id, table_of_particles_name.c_str(),
    				       "xright", &xright, single_element );
    hdf5_status_check( status );
    status = H5LTset_attribute_double( group_id, table_of_particles_name.c_str(),
    				       "ytop", &ytop, single_element );
    hdf5_status_check( status );
    status = H5LTset_attribute_double( group_id, table_of_particles_name.c_str(),
    				       "ybottom", &ybottom, single_element );
    hdf5_status_check( status );
    status = H5LTset_attribute_double( group_id, table_of_particles_name.c_str(),
    				       "zfar", &zfar, single_element );
    hdf5_status_check( status );
    status = H5LTset_attribute_double( group_id, table_of_particles_name.c_str(),
    				       "znear", &znear, single_element );
    hdf5_status_check( status );
    status = H5LTset_attribute_double( group_id, table_of_particles_name.c_str(),
    				       "temperature", &temperature, single_element );
    hdf5_status_check( status );
    status = H5LTset_attribute_double( group_id, table_of_particles_name.c_str(),
    				       "mean_momentum_x", &mean_mom_x, single_element );
    hdf5_status_check( status );
    status = H5LTset_attribute_double( group_id, table_of_particles_name.c_str(),
    				       "mean_momentum_y", &mean_mom_y, single_element );
    hdf5_status_check( status );
    status = H5LTset_attribute_double( group_id, table_of_particles_name.c_str(),
    				       "mean_momentum_z", &mean_mom_z, single_element );
    hdf5_status_check( status );
    status = H5LTset_attribute_double( group_id, table_of_particles_name.c_str(),
    				       "charge", &charge, single_element );
    hdf5_status_check( status );
    status = H5LTset_attribute_double( group_id, table_of_particles_name.c_str(),
    				       "mass", &mass, single_element );
    hdf5_status_check( status );
}


void Particle_source::particle_source_initial_number_of_particles_gt_zero( 
    Config &conf, 
    Source_config_part &src_conf )
{
    check_and_warn_if_not( 
	src_conf.particle_source_initial_number_of_particles > 0,
	"particle_source_initial_number_of_particles <= 0" );
}

void Particle_source::particle_source_particles_to_generate_each_step_ge_zero( 
    Config &conf, 
    Source_config_part &src_conf )
{
    check_and_warn_if_not( 
	src_conf.particle_source_particles_to_generate_each_step >= 0,
	"particle_source_particles_to_generate_each_step < 0" );
}

void Particle_source::particle_source_x_left_ge_zero( 
    Config &conf, 
    Source_config_part &src_conf )
{
    check_and_warn_if_not( 
	src_conf.particle_source_x_left >= 0,
	"particle_source_x_left < 0" );
}

void Particle_source::particle_source_x_left_le_particle_source_x_right( 
    Config &conf, 
    Source_config_part &src_conf )
{
    check_and_warn_if_not( 
	src_conf.particle_source_x_left <= src_conf.particle_source_x_right,
	"particle_source_x_left > particle_source_x_right" );
}

void Particle_source::particle_source_x_right_le_grid_x_size( 
    Config &conf, 
    Source_config_part &src_conf )
{
    check_and_warn_if_not( 
	src_conf.particle_source_x_right <= conf.mesh_config_part.grid_x_size,
	"particle_source_x_right > grid_x_size" );
}

void Particle_source::particle_source_y_bottom_ge_zero( 
    Config &conf, 
    Source_config_part &src_conf )
{
    check_and_warn_if_not( 
	src_conf.particle_source_y_bottom >= 0,
	"particle_source_y_bottom < 0" );
}

void Particle_source::particle_source_y_bottom_le_particle_source_y_top( 
    Config &conf, 
    Source_config_part &src_conf )
{
    check_and_warn_if_not( 
	src_conf.particle_source_y_bottom <= src_conf.particle_source_y_top,
	"particle_source_y_bottom > particle_source_y_top" );
}

void Particle_source::particle_source_y_top_le_grid_y_size( 
    Config &conf, 
    Source_config_part &src_conf )
{
    check_and_warn_if_not( 
	src_conf.particle_source_y_top <= conf.mesh_config_part.grid_y_size,
	"particle_source_y_top > grid_y_size" );
}

void Particle_source::particle_source_z_near_ge_zero( 
    Config &conf, 
    Source_config_part &src_conf )
{
    check_and_warn_if_not( 
	src_conf.particle_source_z_near >= 0,
	"particle_source_z_near < 0" );
}

void Particle_source::particle_source_z_near_le_particle_source_z_far( 
    Config &conf, 
    Source_config_part &src_conf )
{
    check_and_warn_if_not( 
	src_conf.particle_source_z_near <= src_conf.particle_source_z_far,
	"particle_source_z_near > particle_source_z_far" );
}

void Particle_source::particle_source_z_far_le_grid_z_size( 
    Config &conf, 
    Source_config_part &src_conf )
{
    check_and_warn_if_not( 
	src_conf.particle_source_z_far <= conf.mesh_config_part.grid_z_size,
	"particle_source_z_far > grid_z_size" );
}

void Particle_source::particle_source_temperature_gt_zero( 
    Config &conf, 
    Source_config_part &src_conf )
{
    check_and_warn_if_not( 
	src_conf.particle_source_temperature >= 0,
	"particle_source_temperature < 0" );
}

void Particle_source::particle_source_mass_gt_zero( 
    Config &conf, 
    Source_config_part &src_conf )
{
    check_and_warn_if_not( 
	src_conf.particle_source_mass >= 0,
	"particle_source_mass < 0" );
}

void Particle_source::hdf5_status_check( herr_t status )
{
    if( status < 0 ){
	std::cout << "Something went wrong while writing Particle_source "
		  << name << "."
		  << "Aborting." << std::endl;
	exit( EXIT_FAILURE );
    }
}

void check_and_warn_if_not( const bool &should_be, const std::string &message )
{
    if( !should_be ){
	std::cout << "Warning: " + message << std::endl;
    }
    return;
}

Particle_sources_manager::Particle_sources_manager( Config &conf )
{
    for( auto &src_conf : conf.sources_config_part ) {
	sources.emplace_back( conf, src_conf );
    }
}
