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
    unsigned seed = 0 + 1000000*mpi_process_rank;
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

    num_of_particles_for_each_process( &num_of_particles_for_this_proc,
				       num_of_particles );
    populate_vec_of_ids( vec_of_ids, num_of_particles_for_this_proc ); 
    for ( int i = 0; i < num_of_particles_for_this_proc; i++ ) {
	pos = uniform_position_in_cube( xleft, ytop, znear,
					xright, ybottom, zfar,
					rnd_gen );
	mom = maxwell_momentum_distr( mean_momentum, temperature, mass, rnd_gen );
	particles.emplace_back( vec_of_ids[i], charge, mass, pos, mom );
    }
}

void Particle_source::num_of_particles_for_each_process(
    int *num_of_particles_for_this_proc,
    int num_of_particles )
{
    int rest;
    int mpi_n_of_proc, mpi_process_rank;
    MPI_Comm_size( MPI_COMM_WORLD, &mpi_n_of_proc );
    MPI_Comm_rank( MPI_COMM_WORLD, &mpi_process_rank );    
    
    *num_of_particles_for_this_proc = num_of_particles / mpi_n_of_proc;
    rest = num_of_particles % mpi_n_of_proc;
    if( mpi_process_rank < rest ){
	(*num_of_particles_for_this_proc)++;
	// Processes with lesser ranks will accumulate
	// more particles.
	// This seems unessential.
    }    
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

void Particle_source::write_to_file_iostream( std::ofstream &output_file )
{
    std::cout << "Source name = " << name << ", "
	      << "number of particles = " << particles.size() 
	      << std::endl;
    output_file << "Source name = " << name << std::endl;
    output_file << "Total number of particles = " << particles.size() << std::endl;
    output_file << "id, charge, mass, position(x,y,z), momentum(px,py,pz)" << std::endl;
    output_file.fill(' ');
    output_file.setf( std::ios::scientific );
    output_file.precision( 3 );    
    output_file.setf( std::ios::right );
    for ( auto &p : particles ) {	
	output_file << std::setw(10) << std::left << p.id
		    << std::setw(12) << p.charge
		    << std::setw(12) << p.mass
		    << std::setw(12) << vec3d_x( p.position )
		    << std::setw(12) << vec3d_y( p.position )
		    << std::setw(12) << vec3d_z( p.position )
		    << std::setw(12) << vec3d_x( p.momentum )
		    << std::setw(12) << vec3d_y( p.momentum )
		    << std::setw(12) << vec3d_z( p.momentum )
		    << std::endl;
    }
    return;
}

void Particle_source::write_to_file_hdf5( hid_t group_id )
{
    std::cout << "Source name = " << name << ", "
	      << "number of particles = " << particles.size() 
	      << std::endl;
    std::string table_of_particles_name = name;

    write_hdf5_particles( group_id, table_of_particles_name );
    write_hdf5_source_parameters( group_id, table_of_particles_name );
    
    return;
}

void Particle_source::write_hdf5_particles( hid_t group_id, std::string table_of_particles_name )
{
    // todo: remove
    int mpi_n_of_proc, mpi_process_rank;
    MPI_Comm_size( MPI_COMM_WORLD, &mpi_n_of_proc );
    MPI_Comm_rank( MPI_COMM_WORLD, &mpi_process_rank );    
    //
    
    hid_t filespace, memspace, dset;
    hid_t compound_type_for_mem, compound_type_for_file; 
    hid_t plist_id;
    herr_t status;
    int rank = 1;
    hsize_t dims[rank], subset_dims[rank], subset_offset[rank];
    dims[0] = particles.size();    
    
    // todo: dst_buf should be removed.
    // currently it is used to avoid any problems of
    // working with Particles class, which is a C++ class
    // and not a plain C datastructure
    HDF5_buffer_for_Particle *dst_buf = new HDF5_buffer_for_Particle[ particles.size() ];
    for( unsigned int i = 0; i < particles.size(); i++ ){
	dst_buf[i].id = particles[i].id;
	dst_buf[i].charge = particles[i].charge;
	dst_buf[i].mass = particles[i].mass;
	dst_buf[i].position = particles[i].position;
	dst_buf[i].momentum = particles[i].momentum;
	dst_buf[i].mpi_proc_rank = mpi_process_rank;
    }	
    
    compound_type_for_mem = HDF5_buffer_for_Particle_compound_type_for_memory();
    compound_type_for_file = HDF5_buffer_for_Particle_compound_type_for_file();
    plist_id = H5Pcreate( H5P_DATASET_XFER );
    H5Pset_dxpl_mpio( plist_id, H5FD_MPIO_COLLECTIVE );

    subset_dims[0] = n_of_elements_to_write_for_each_process_for_1d_dataset( dims[0] );
    subset_offset[0] = data_offset_for_each_process_for_1d_dataset( dims[0] );

    // todo: remove
    std::cout << "particles "
	      << "total = " << particles.size() << " "
	      << "proc_n = " << mpi_process_rank << " "
	      << "count = " << subset_dims[0] << " "
	      << "offset = " << subset_offset[0] << std::endl;
    //
    
    memspace = H5Screate_simple( rank, subset_dims, NULL );
    filespace = H5Screate_simple( rank, dims, NULL );
    H5Sselect_hyperslab( filespace, H5S_SELECT_SET, subset_offset, NULL, subset_dims, NULL );
    
    dset = H5Dcreate( group_id, ("./" + table_of_particles_name).c_str(),
		      compound_type_for_file, filespace,
		      H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT );
    status = H5Dwrite( dset, compound_type_for_mem, memspace, filespace, plist_id, dst_buf );
    status = H5Dclose( dset );

    status = H5Sclose( filespace );
    status = H5Sclose( memspace );
    status = H5Pclose( plist_id );
    status = H5Tclose( compound_type_for_file );
    status = H5Tclose( compound_type_for_mem );	
    delete[] dst_buf;
}

int Particle_source::n_of_elements_to_write_for_each_process_for_1d_dataset( int total_elements )
{
    int mpi_n_of_proc, mpi_process_rank;
    MPI_Comm_size( MPI_COMM_WORLD, &mpi_n_of_proc );
    MPI_Comm_rank( MPI_COMM_WORLD, &mpi_process_rank );    

    int n_of_elements_for_process = total_elements / mpi_n_of_proc;
    int rest = total_elements % mpi_n_of_proc;
    if( mpi_process_rank < rest ){
	n_of_elements_for_process++;
    }

    return n_of_elements_for_process;
}

int Particle_source::data_offset_for_each_process_for_1d_dataset( int total_elements )
{
    int mpi_n_of_proc, mpi_process_rank;
    MPI_Comm_size( MPI_COMM_WORLD, &mpi_n_of_proc );
    MPI_Comm_rank( MPI_COMM_WORLD, &mpi_process_rank );    

    // todo: it is simpler to calclulate offset directly than
    // to perform MPI broadcast of n_of_elements_for_each_proc. 
    int offset;
    int min_n_of_elements_for_process = total_elements / mpi_n_of_proc;
    int max_n_of_elements_for_process = min_n_of_elements_for_process + 1;
    int rest = total_elements % mpi_n_of_proc;

    if( mpi_process_rank < rest ){
	offset = mpi_process_rank * max_n_of_elements_for_process;
    } else {
	offset = rest * max_n_of_elements_for_process +
	    ( mpi_process_rank - rest ) * min_n_of_elements_for_process;
    }

    return offset;
}


void Particle_source::write_hdf5_source_parameters( hid_t group_id,
						    std::string table_of_particles_name )
{
    int single_element = 1;
    double mean_mom_x = vec3d_x( mean_momentum );
    double mean_mom_y = vec3d_y( mean_momentum );
    double mean_mom_z = vec3d_z( mean_momentum );

    H5LTset_attribute_double( group_id, table_of_particles_name.c_str(), "xleft", &xleft, single_element );
    H5LTset_attribute_double( group_id, table_of_particles_name.c_str(), "xright", &xright, single_element );
    H5LTset_attribute_double( group_id, table_of_particles_name.c_str(), "ytop", &ytop, single_element );
    H5LTset_attribute_double( group_id, table_of_particles_name.c_str(), "ybottom", &ybottom, single_element );
    H5LTset_attribute_double( group_id, table_of_particles_name.c_str(), "zfar", &zfar, single_element );
    H5LTset_attribute_double( group_id, table_of_particles_name.c_str(), "znear", &znear, single_element );
    H5LTset_attribute_double( group_id, table_of_particles_name.c_str(), "temperature", &temperature, single_element );
    H5LTset_attribute_double( group_id, table_of_particles_name.c_str(),
			      "mean_momentum_x", &mean_mom_x, single_element );
    H5LTset_attribute_double( group_id, table_of_particles_name.c_str(),
			      "mean_momentum_y", &mean_mom_y, single_element );
    H5LTset_attribute_double( group_id, table_of_particles_name.c_str(),
			      "mean_momentum_z", &mean_mom_z, single_element );    
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
