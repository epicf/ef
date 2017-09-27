#include "domain.h"

// Domain print
std::string construct_output_filename( const std::string output_filename_prefix, 
				       const int current_time_step,
				       const std::string output_filename_suffix );

//
// Domain initialization
//

Domain::Domain( Config &conf ) :
    time_grid( conf ),
    spat_mesh( conf ),
    inner_regions( conf, spat_mesh ),
    particle_to_mesh_map(),
    field_solver( spat_mesh, inner_regions ),
    particle_sources( conf ),
    external_fields( conf ),
    particle_interaction_model( conf )
{
    output_filename_prefix = conf.output_filename_config_part.output_filename_prefix;
    output_filename_suffix = conf.output_filename_config_part.output_filename_suffix;    
    return;
}

Domain::Domain( hid_t h5file_id ) :
    time_grid( H5Gopen( h5file_id, "/Time_grid", H5P_DEFAULT ) ),
    spat_mesh( H5Gopen( h5file_id, "/Spatial_mesh", H5P_DEFAULT ) ),
    inner_regions( H5Gopen( h5file_id, "/Inner_regions", H5P_DEFAULT ), spat_mesh ),
    particle_to_mesh_map(),
    field_solver( spat_mesh, inner_regions ),
    particle_sources( H5Gopen( h5file_id, "/Particle_sources", H5P_DEFAULT ) ),
    external_fields( H5Gopen( h5file_id, "/External_fields", H5P_DEFAULT ) ),
    particle_interaction_model(
	H5Gopen( h5file_id, "/Particle_interaction_model", H5P_DEFAULT ) )
{
    return;
}


//
// Pic simulation 
//

void Domain::start_pic_simulation()
{
    // fields in domain without any particles
    eval_and_write_fields_without_particles();
    // generate particles and write initial step
    prepare_boris_integration();
    write_step_to_save();
    // run simulation
    run_pic();

    return;
}

void Domain::continue_pic_simulation()
{
    run_pic();
    return;
}


void Domain::run_pic()
{
    int mpi_process_rank;
    MPI_Comm_rank( PETSC_COMM_WORLD, &mpi_process_rank );
    
    int total_time_iterations, current_node;
    total_time_iterations = time_grid.total_nodes - 1;
    current_node = time_grid.current_node;

    for ( int i = current_node; i < total_time_iterations; i++ ){
	if ( mpi_process_rank == 0 ){
	    std::cout << "Time step from " << i << " to " << i+1
		      << " of " << total_time_iterations << std::endl;
	}
    	advance_one_time_step();
    	write_step_to_save();
    }

    return;
}

void Domain::prepare_boris_integration()
{
    if ( particle_interaction_model.noninteracting ){
	shift_velocities_half_time_step_back();
    } else if ( particle_interaction_model.pic ){
	eval_charge_density();
	eval_potential_and_fields();
	shift_velocities_half_time_step_back();
    }
    return;
}

void Domain::advance_one_time_step()
{    
    if ( particle_interaction_model.noninteracting ){
	push_particles();
	apply_domain_constrains();
	update_time_grid();
    } else if ( particle_interaction_model.pic ){
	push_particles();
	apply_domain_constrains();
	eval_charge_density();
	eval_potential_and_fields();
	update_time_grid();
    }
    return;
}

void Domain::eval_charge_density()
{
    spat_mesh.clear_old_density_values();    
    particle_to_mesh_map.weight_particles_charge_to_mesh( spat_mesh, particle_sources );
    
    return;
}


void Domain::eval_potential_and_fields()
{
    field_solver.eval_potential( spat_mesh, inner_regions );
    field_solver.eval_fields_from_potential( spat_mesh );
    return;
}

void Domain::push_particles()
{
    boris_integration();
    return;
}

void Domain::apply_domain_constrains()
{
    // First generate then remove.
    // This allows for overlap of source and inner region.
    generate_new_particles();

    apply_domain_boundary_conditions();
    remove_particles_inside_inner_regions();    
    return;
}

//
// Push particles
//

void Domain::boris_integration()
{  
    double dt = time_grid.time_step_size;

    update_momentum( dt );
    update_position( dt );
    return;
}

void Domain::shift_velocities_half_time_step_back()
{
    double minus_half_dt = -time_grid.time_step_size / 2;
    Vec3d el_field, mgn_field, pic_el_field;
    Vec3d total_el_field, total_mgn_field;    
    Vec3d h, s, u, u_quote, v_current, half_el_force;
    Vec3d dp;
    double q_quote;

    for( auto &src : particle_sources.sources ) {
	for( auto &p : src.particles ) {
	    if ( !p.momentum_is_half_time_step_shifted ){
		total_el_field = vec3d_zero();
		for( auto &f : external_fields.electric ) {
		    el_field = f.field_at_particle_position( p, time_grid.current_time );
		    total_el_field = vec3d_add( total_el_field, el_field );
		}
		pic_el_field = particle_to_mesh_map.field_at_particle_position(
		    spat_mesh, p );
		total_el_field = vec3d_add( total_el_field, pic_el_field );

		total_mgn_field = vec3d_zero();
		for( auto &f : external_fields.magnetic ) {		
		    mgn_field = f.field_at_particle_position( p, time_grid.current_time );
		    total_mgn_field = vec3d_add( total_mgn_field, mgn_field );
		}
	    
		if ( external_fields.magnetic.empty() ){
		    dp = vec3d_times_scalar( total_el_field,
					     p.charge * minus_half_dt );
		    p.momentum = vec3d_add( p.momentum, dp );
		} else {			    
		    q_quote = minus_half_dt * p.charge / p.mass / 2.0;
		    half_el_force = vec3d_times_scalar( total_el_field, q_quote );
		    v_current = vec3d_times_scalar( p.momentum, 1.0 / p.mass );
		    u = vec3d_add( v_current, half_el_force );		
		    h = vec3d_times_scalar(
			total_mgn_field,
			q_quote / physconst_speed_of_light );
		    s = vec3d_times_scalar(
			h,
			2.0 / ( 1.0 + vec3d_dot_product( h, h ) ) );
		    u_quote = vec3d_add(
			u,
			vec3d_cross_product(
			    vec3d_add( u, vec3d_cross_product( u, h ) ),
			    s ) );
		    p.momentum = vec3d_times_scalar(
			vec3d_add( u_quote, half_el_force ),
			p.mass );
		}
		p.momentum_is_half_time_step_shifted = true;
	    }
	}
    }
    return;
}

void Domain::update_momentum( double dt )
{
    Vec3d el_field, mgn_field, pic_el_field;
    Vec3d total_el_field, total_mgn_field;
    Vec3d h, s, u, u_quote, v_current, half_el_force;
    Vec3d dp;
    double q_quote;

    for( auto &src : particle_sources.sources ) {
	for( auto &p : src.particles ) {
	    total_el_field = vec3d_zero();
	    for( auto &f : external_fields.electric ) {
		el_field = f.field_at_particle_position( p, time_grid.current_time );
		total_el_field = vec3d_add( total_el_field, el_field );
	    }
	    pic_el_field = particle_to_mesh_map.field_at_particle_position( spat_mesh, p );
	    total_el_field = vec3d_add( total_el_field, pic_el_field );

	    total_mgn_field = vec3d_zero();
	    for( auto &f : external_fields.magnetic ) {		
		mgn_field = f.field_at_particle_position( p, time_grid.current_time );
		total_mgn_field = vec3d_add( total_mgn_field, mgn_field );
	    }
	    
	    if ( external_fields.magnetic.empty() ){
		dp = vec3d_times_scalar( total_el_field,
					 p.charge * time_grid.time_step_size );
		p.momentum = vec3d_add( p.momentum, dp );
	    } else {			    
		q_quote = time_grid.time_step_size * p.charge / p.mass / 2.0;
		half_el_force = vec3d_times_scalar( total_el_field, q_quote );
		v_current = vec3d_times_scalar( p.momentum, 1.0 / p.mass );
		u = vec3d_add( v_current, half_el_force );		
		h = vec3d_times_scalar(
		    total_mgn_field,
		    q_quote / physconst_speed_of_light );
		s = vec3d_times_scalar(
		    h,
		    2.0 / ( 1.0 + vec3d_dot_product( h, h ) ) );
		u_quote = vec3d_add(
		    u,
		    vec3d_cross_product(
			vec3d_add( u, vec3d_cross_product( u, h ) ),
			s ) );
		p.momentum = vec3d_times_scalar(
		    vec3d_add( u_quote, half_el_force ),
		    p.mass );
	    }
	}
    }
    return;
}

void Domain::update_position( double dt )
{
    particle_sources.update_particles_position( dt );
    return;
}

//
// Apply domain constrains
//

void Domain::apply_domain_boundary_conditions()
{
    for( auto &src : particle_sources.sources ) {
    	auto remove_starting_from = std::remove_if( 
    	    std::begin( src.particles ), 
    	    std::end( src.particles ), 
    	    [this]( Particle &p ){ return out_of_bound(p); } ); 
    	// cout << "Out of bound from " << src.name << ":" << " "
    	//      << std::end( src.particles ) - remove_starting_from << std::endl;
    	src.particles.erase(
    	    remove_starting_from,
    	    std::end( src.particles ) );
    }

    return;
}

void Domain::remove_particles_inside_inner_regions()
{
    for( auto &src : particle_sources.sources ) {
	auto remove_starting_from = std::remove_if( 
	    std::begin( src.particles ), 
	    std::end( src.particles ), 
	    [this]( Particle &p ){
		return inner_regions.check_if_particle_inside_and_count_charge( p );
	    } ); 
	inner_regions.sync_absorbed_charge_and_particles_across_proc();
	src.particles.erase( remove_starting_from, std::end( src.particles ) );
    }
    return;
}

bool Domain::out_of_bound( const Particle &p )
{
    double x = vec3d_x( p.position );
    double y = vec3d_y( p.position );
    double z = vec3d_z( p.position );
    bool out;
    
    out = 
	( x >= spat_mesh.x_volume_size ) || ( x <= 0 ) ||
	( y >= spat_mesh.y_volume_size ) || ( y <= 0 ) ||
	( z >= spat_mesh.z_volume_size ) || ( z <= 0 ) ;
	
    return out;

}

void Domain::generate_new_particles()
{
    particle_sources.generate_each_step();
    shift_velocities_half_time_step_back();
    return;
}


//
// Update time grid
//

void Domain::update_time_grid()
{
    time_grid.update_to_next_step();
    return;
}

//
// Write domain to file
//

void Domain::write_step_to_save()
{
    int current_step = time_grid.current_node;
    int step_to_save = time_grid.node_to_save;
    if ( ( current_step % step_to_save ) == 0 ){	
	write();
    }
    return;
}

void Domain::write()
{    
    herr_t status;    
    std::string file_name_to_write;
    
    file_name_to_write = construct_output_filename( output_filename_prefix, 
						    time_grid.current_node,
						    output_filename_suffix  );
    hid_t plist_id;
    plist_id = H5Pcreate( H5P_FILE_ACCESS ); hdf5_status_check( plist_id );
    status = H5Pset_fapl_mpio( plist_id, MPI_COMM_WORLD, MPI_INFO_NULL );
    hdf5_status_check( status );

    hid_t output_file = H5Fcreate( file_name_to_write.c_str(),
				   H5F_ACC_TRUNC, H5P_DEFAULT, plist_id );
    if ( negative( output_file ) ) {
	std::cout << "Error: can't open file \'" 
		  << file_name_to_write 
		  << "\' to save results of simulation!" 
		  << std::endl;
	std::cout << "Recheck \'output_filename_prefix\' key in config file." 
		  << std::endl;
	std::cout << "Make sure the directory you want to save to exists." 
		  << std::endl;
	exit( EXIT_FAILURE );
    }

    int mpi_process_rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &mpi_process_rank );    
    if( mpi_process_rank == 0 ){    
	std::cout << "Writing step " << time_grid.current_node 
		  << " to file " << file_name_to_write << std::endl;
    }

    time_grid.write_to_file( output_file );
    spat_mesh.write_to_file( output_file );
    particle_sources.write_to_file( output_file );
    inner_regions.write_to_file( output_file );
    external_fields.write_to_file( output_file );
    particle_interaction_model.write_to_file( output_file );

    status = H5Pclose( plist_id ); hdf5_status_check( status );
    status = H5Fclose( output_file ); hdf5_status_check( status );

    return;
}


std::string construct_output_filename( const std::string output_filename_prefix, 
				       const int current_time_step,
				       const std::string output_filename_suffix )
{    
    std::stringstream step_string;
    step_string << std::setfill('0') << std::setw(7) <<  current_time_step;

    std::string filename;
    filename = output_filename_prefix + 
	step_string.str() + 
	output_filename_suffix;
    return filename;
}

//
// Free domain
//

Domain::~Domain()
{
    std::cout << "TODO: free domain.\n";
    return;
}

//
// Various functions
//
void Domain::set_output_filename_prefix_and_suffix( std::string prefix, std::string suffix )
{
    output_filename_prefix = prefix;
    output_filename_suffix = suffix;
}


void Domain::print_particles() 
{
    particle_sources.print_particles();
    return;
}

void Domain::eval_and_write_fields_without_particles()
{
    herr_t status;

    spat_mesh.clear_old_density_values();
    eval_potential_and_fields();

    std::string file_name_to_write;
    
    file_name_to_write = output_filename_prefix + 
	"fieldsWithoutParticles" + 
	output_filename_suffix;

    hid_t plist_id;
    plist_id = H5Pcreate( H5P_FILE_ACCESS ); hdf5_status_check( plist_id );
    status = H5Pset_fapl_mpio( plist_id, MPI_COMM_WORLD, MPI_INFO_NULL );
    hdf5_status_check( status );

    hid_t output_file = H5Fcreate( file_name_to_write.c_str(),
				   H5F_ACC_TRUNC, H5P_DEFAULT, plist_id );
    if ( negative( output_file ) ) {
	std::cout << "Error: can't open file \'" 
		  << file_name_to_write 
		  << "\' to save results of initial field calculation!" 
		  << std::endl;
	std::cout << "Recheck \'output_filename_prefix\' key in config file." 
		  << std::endl;
	std::cout << "Make sure the directory you want to save to exists." 
		  << std::endl;
	exit( EXIT_FAILURE );
    }

    int mpi_process_rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &mpi_process_rank );    
    if( mpi_process_rank == 0 ){    
	std::cout << "Writing initial fields" << " "
		  << "to file " << file_name_to_write << std::endl;
    }
    
    spat_mesh.write_to_file( output_file );
    external_fields.write_to_file( output_file );
    inner_regions.write_to_file( output_file );

    status = H5Pclose( plist_id ); hdf5_status_check( status );
    status = H5Fclose( output_file ); hdf5_status_check( status );

    return;
}

bool Domain::negative( hid_t hdf5_id )
{
    return hdf5_id < 0;
}

void Domain::hdf5_status_check( herr_t status )
{
    if( status < 0 ){
	std::cout << "Something went wrong while writing root group of HDF5 file. Aborting."
		  << std::endl;
	exit( EXIT_FAILURE );
    }
}
