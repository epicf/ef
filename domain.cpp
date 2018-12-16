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
    time_grid( H5Gopen( h5file_id, "/TimeGrid", H5P_DEFAULT ) ),
    spat_mesh( H5Gopen( h5file_id, "/SpatialMesh", H5P_DEFAULT ) ),
    inner_regions( H5Gopen( h5file_id, "/InnerRegions", H5P_DEFAULT ), spat_mesh ),
    particle_to_mesh_map(),
    field_solver( spat_mesh, inner_regions ),
    particle_sources( H5Gopen( h5file_id, "/ParticleSources", H5P_DEFAULT ) ),
    external_fields( H5Gopen( h5file_id, "/ExternalFields", H5P_DEFAULT ) ),
    particle_interaction_model(
        H5Gopen( h5file_id, "/ParticleInteractionModel", H5P_DEFAULT ) )
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
    int total_time_iterations, current_node;
    total_time_iterations = time_grid.total_nodes - 1;
    current_node = time_grid.current_node;

    for ( int i = current_node; i < total_time_iterations; i++ ){
        std::cout << "Time step from " << i << " to " << i+1
                  << " of " << total_time_iterations << std::endl;
        advance_one_time_step();
        write_step_to_save();
    }

    return;
}

void Domain::prepare_boris_integration()
{
    if ( particle_interaction_model.pic ){
        eval_charge_density();
        eval_potential_and_fields();
    }
    shift_new_particles_velocities_half_time_step_back();
}

void Domain::advance_one_time_step()
{
    push_particles();
    apply_domain_constrains();
    if ( particle_interaction_model.pic ){
        eval_charge_density();
        eval_potential_and_fields();
    }
    update_time_grid();
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
    field_solver.eval_potential(inner_regions );
    field_solver.eval_fields_from_potential();
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
    update_momentum( time_grid.time_step_size );
    update_position( time_grid.time_step_size );
    return;
}

void Domain::shift_new_particles_velocities_half_time_step_back()
{
    double minus_half_dt = -time_grid.time_step_size / 2.0;
    Vec3d total_el_field, total_mgn_field;
    unsigned int source_idx, particle_idx;

    for( source_idx = 0;
         source_idx < particle_sources.sources.size();
         source_idx++ ){
        auto &src = particle_sources.sources[ source_idx ];
        for( particle_idx = 0;
             particle_idx < src.particles.size();
             particle_idx++ ){
            auto &p = src.particles[ particle_idx ];
            if ( !p.momentum_is_half_time_step_shifted ){
                total_el_field = compute_electric_field_at_particle_position(
                    p, particle_idx, source_idx );
                total_mgn_field = compute_magnetic_field_at_particle_position( p );
                //
                if( external_fields.magnetic.empty() ){
                    boris_update_particle_momentum_no_mgn_field(
                        p, minus_half_dt, total_el_field );
                } else {
                    boris_update_particle_momentum(
                        p, minus_half_dt, total_el_field, total_mgn_field );
                }
                p.momentum_is_half_time_step_shifted = true;
            }
        }
    }
    return;
}

void Domain::update_momentum( double dt )
{
    Vec3d total_el_field, total_mgn_field;
    unsigned int source_idx, particle_idx;

    for( source_idx = 0;
         source_idx < particle_sources.sources.size();
         source_idx++ ){
        auto &src = particle_sources.sources[ source_idx ];
        for( particle_idx = 0;
             particle_idx < src.particles.size();
             particle_idx++ ){
            auto &p = src.particles[ particle_idx ];
            total_el_field = compute_electric_field_at_particle_position(
                p, particle_idx, source_idx );
            total_mgn_field = compute_magnetic_field_at_particle_position( p );
            //
            if ( external_fields.magnetic.empty() ){
                boris_update_particle_momentum_no_mgn_field( p, dt, total_el_field );
            } else {
                boris_update_particle_momentum( p, dt, total_el_field, total_mgn_field );
            }
        }
    }
    return;
}


Vec3d Domain::compute_electric_field_at_particle_position(
    Particle &particle, unsigned int particle_idx, unsigned int source_idx )
{
    Vec3d ext_el_field, bin_el_field, mesh_el_field, mesh_and_pic_el_field;
    Vec3d total_el_field;

    bool inner_regs = ! inner_regions.regions.empty();
    bool gradient = ! spat_mesh.is_potential_equal_on_boundaries();

    total_el_field = vec3d_zero();
    for( auto &f : external_fields.electric ) {
        ext_el_field = f.field_at_particle_position( particle, time_grid.current_time );
        total_el_field = vec3d_add( total_el_field, ext_el_field );
    }
    if ( particle_interaction_model.noninteracting ){
        if ( inner_regs or gradient ){
            mesh_el_field = particle_to_mesh_map.field_at_particle_position(
                spat_mesh, particle);
            total_el_field = vec3d_add( total_el_field, mesh_el_field );
        }
    } else if ( particle_interaction_model.binary ){
        bin_el_field = binary_field_at_particle_position(
            particle, particle_idx, source_idx);
        total_el_field = vec3d_add( total_el_field, bin_el_field );
        if ( inner_regs or gradient ){
            mesh_el_field = particle_to_mesh_map.field_at_particle_position(
                spat_mesh, particle);
            total_el_field = vec3d_add( total_el_field, mesh_el_field );
        }
    } else if ( particle_interaction_model.pic ){
        mesh_and_pic_el_field = particle_to_mesh_map.field_at_particle_position(
            spat_mesh, particle );
        total_el_field = vec3d_add( total_el_field, mesh_and_pic_el_field );
    }
    return total_el_field;
}

Vec3d Domain::compute_magnetic_field_at_particle_position( Particle &particle )
{
    Vec3d ext_mgn_field;
    Vec3d total_mgn_field;

    total_mgn_field = vec3d_zero();
    for( auto &f : external_fields.magnetic ) {
        ext_mgn_field = f.field_at_particle_position( particle, time_grid.current_time );
        total_mgn_field = vec3d_add( total_mgn_field, ext_mgn_field );
    }
    return total_mgn_field;
}


Vec3d Domain::binary_field_at_particle_position(
    Particle &particle, unsigned int particle_idx, unsigned int source_idx )
{
    Vec3d bin_force = vec3d_zero();
    unsigned int src_iter, part_iter;
    //Particle tmp;

    for( src_iter = 0; src_iter < particle_sources.sources.size(); src_iter++ ){
        auto &src = particle_sources.sources[ src_iter ];
        if ( source_idx != src_iter ){
            for( auto &p : src.particles ){
                bin_force = vec3d_add( bin_force, p.field_at_point( particle.position ));
            }
        } else if ( src.particles.size() > 1 ){
            std::swap( src.particles[0], src.particles[particle_idx] );
            for( part_iter = 1;
                 part_iter < src.particles.size();
                 part_iter++){
                auto &p = src.particles[part_iter];
                bin_force = vec3d_add( bin_force,
                                       p.field_at_point( src.particles[0].position ));
            }
            // swap particles back so that loop in 'prepare_boris/update_momentum'
            // is not disturbed
            std::swap( src.particles[0], src.particles[particle_idx] );
        }
    }
    return bin_force;
}

void Domain::boris_update_particle_momentum_no_mgn_field(
    Particle &p, double dt, Vec3d total_el_field )
{
    Vec3d dp;

    dp = vec3d_times_scalar( total_el_field, p.charge * dt );
    p.momentum = vec3d_add( p.momentum, dp );
}


void Domain::boris_update_particle_momentum(
    Particle &p, double dt,
    Vec3d total_el_field, Vec3d total_mgn_field )
{
    Vec3d h, s, u, u_quote, v_current, half_el_force;
    double q_quote;

    q_quote = dt * p.charge / p.mass / 2.0;
    half_el_force = vec3d_times_scalar( total_el_field, q_quote );
    v_current = vec3d_times_scalar( p.momentum, 1.0 / p.mass );
    u = vec3d_add( v_current, half_el_force );
    h = vec3d_times_scalar( total_mgn_field, q_quote / physconst_speed_of_light );
    s = vec3d_times_scalar( h,
                            2.0 / ( 1.0 + vec3d_dot_product( h, h ) ) );
    u_quote = vec3d_add(
        u,
        vec3d_cross_product(
            vec3d_add( u, vec3d_cross_product( u, h ) ),
            s ) );
    p.momentum = vec3d_times_scalar( vec3d_add( u_quote, half_el_force ),
                                     p.mass );
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
        ( x >= spat_mesh.volume_size.x ) || ( x <= 0 ) ||
        ( y >= spat_mesh.volume_size.y ) || ( y <= 0 ) ||
        ( z >= spat_mesh.volume_size.z ) || ( z <= 0 ) ;

    return out;

}

void Domain::generate_new_particles()
{
    particle_sources.generate_each_step();
    shift_new_particles_velocities_half_time_step_back();
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

    hid_t output_file = H5Fcreate( file_name_to_write.c_str(),
                                   H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT );
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

    std::cout << "Writing step " << time_grid.current_node
              << " to file " << file_name_to_write << std::endl;

    time_grid.write_to_file( output_file );
    spat_mesh.write_to_file( output_file );
    particle_sources.write_to_file( output_file );
    inner_regions.write_to_file( output_file );
    external_fields.write_to_file( output_file );
    particle_interaction_model.write_to_file( output_file );

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

    hid_t output_file = H5Fcreate( file_name_to_write.c_str(),
                                   H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT );
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

    std::cout << "Writing initial fields" << " "
              << "to file " << file_name_to_write << std::endl;

    spat_mesh.write_to_file( output_file );
    external_fields.write_to_file( output_file );
    inner_regions.write_to_file( output_file );

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
