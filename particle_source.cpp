#include "particle_source.h"

void check_and_exit_if_not( const bool &should_be, const std::string &message );

Particle_source::Particle_source( Config *conf )
{
    check_correctness_of_related_config_fields( conf );
    set_parameters_from_config( conf );
    generate_initial_particles();
}

void Particle_source::check_correctness_of_related_config_fields( Config *conf )
{
    particle_source_initial_number_of_particles_gt_zero( conf );
    particle_source_particles_to_generate_each_step_ge_zero( conf );
    particle_source_x_left_ge_zero( conf );
    particle_source_x_left_le_particle_source_x_right( conf );
    particle_source_x_right_le_grid_x_size( conf );
    particle_source_y_bottom_ge_zero( conf );
    particle_source_y_bottom_le_particle_source_y_top( conf );
    particle_source_y_top_le_grid_y_size( conf );
    particle_source_temperature_gt_zero( conf );
    particle_source_mass_gt_zero( conf );
}

void Particle_source::set_parameters_from_config( Config *conf )
{
    initial_number_of_particles = conf->sources_config_part.at(0).particle_source_initial_number_of_particles;
    particles_to_generate_each_step = conf->sources_config_part.at(0).particle_source_particles_to_generate_each_step;
    xleft = conf->sources_config_part.at(0).particle_source_x_left;
    xright = conf->sources_config_part.at(0).particle_source_x_right;
    ytop = conf->sources_config_part.at(0).particle_source_y_top;
    ybottom = conf->sources_config_part.at(0).particle_source_y_bottom;
    temperature = conf->sources_config_part.at(0).particle_source_temperature;
    charge = conf->sources_config_part.at(0).particle_source_charge;
    mass = conf->sources_config_part.at(0).particle_source_mass;    
    // Random number generator
    unsigned seed = 0;
    rnd_gen = std::default_random_engine( seed );
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
    Vec2d pos, mom;
    int id = 0;
                
    for ( int i = 0; i < num_of_particles; i++ ) {
	id = generate_particle_id( i );
	pos = uniform_position_in_rectangle( xleft, ytop, xright, ybottom, rnd_gen );
	mom = maxwell_momentum_distr( temperature, mass, rnd_gen );
	particles.emplace_back( id, charge, mass, pos, mom );
    }

}

int Particle_source::generate_particle_id( const int number )
{    
    // Preserve max id between calls to generator.
    static int last_id = 0;
    
    return last_id++;
}

Vec2d Particle_source::uniform_position_in_rectangle( const double xleft,  const double ytop,
						      const double xright, const double ybottom,
						      std::default_random_engine &rnd_gen )
{
    return vec2d_init( random_in_range( xleft, xright, rnd_gen ), 
		       random_in_range( ybottom, ytop, rnd_gen ) );
}

double Particle_source::random_in_range( const double low, const double up, 
					 std::default_random_engine &rnd_gen )
{
    std::uniform_real_distribution<double> uniform_distr( low, up );
    return uniform_distr( rnd_gen );
}

Vec2d Particle_source::maxwell_momentum_distr( const double temperature, const double mass, 
					       std::default_random_engine &rnd_gen )
{    
    double maxwell_gauss_std_mean = 0.0;
    double maxwell_gauss_std_dev = sqrt( mass * temperature );
    std::normal_distribution<double> normal_distr( maxwell_gauss_std_mean, maxwell_gauss_std_dev );

    Vec2d mom;
    mom = vec2d_init( normal_distr( rnd_gen ),
		      normal_distr( rnd_gen ) );		     
    mom = vec2d_times_scalar( mom, 1.0 ); // recheck
    return mom;
}

void Particle_source::print_all()
{
    for ( auto& p : particles  ) {
	p.print();
    }
    return;
}

void Particle_source::write_to_file( FILE *f )
{
    printf( "Number of particles  = %d \n", particles.size() );
    fprintf( f, "### Particles\n" );
    fprintf( f, "Total number of particles = %d\n", particles.size() );    
    fprintf( f, "id \t   charge      mass \t position(x,y) \t\t momentum(px,py) \n");
    for ( auto &p : particles ) {	
	fprintf( f, 
		 "%-10d %-10.3f %-10.3f %-10.3f %-10.3f  %-10.3f %-10.3f \n", 
		 p.id, p.charge, p.mass, 
		 vec2d_x( p.position ), vec2d_y( p.position ), 
		 vec2d_x( p.momentum ), vec2d_y( p.momentum ) );
    }
    return;
}



void Particle_source::particle_source_initial_number_of_particles_gt_zero( Config *conf )
{
    check_and_exit_if_not( 
	conf->sources_config_part.at(0).particle_source_initial_number_of_particles > 0,
	"particle_source_initial_number_of_particles <= 0" );
}

void Particle_source::particle_source_particles_to_generate_each_step_ge_zero( Config *conf )
{
    check_and_exit_if_not( 
	conf->sources_config_part.at(0).particle_source_particles_to_generate_each_step >= 0,
	"particle_source_particles_to_generate_each_step < 0" );
}

void Particle_source::particle_source_x_left_ge_zero( Config *conf )
{
    check_and_exit_if_not( 
	conf->sources_config_part.at(0).particle_source_x_left >= 0,
	"particle_source_x_left < 0" );
}

void Particle_source::particle_source_x_left_le_particle_source_x_right( Config *conf )
{
    check_and_exit_if_not( 
	conf->sources_config_part.at(0).particle_source_x_left <= conf->sources_config_part.at(0).particle_source_x_right,
	"particle_source_x_left > particle_source_x_right" );
}

void Particle_source::particle_source_x_right_le_grid_x_size( Config *conf )
{
    check_and_exit_if_not( 
	conf->sources_config_part.at(0).particle_source_x_right <= conf->mesh_config_part.grid_x_size,
	"particle_source_x_right > grid_x_size" );
}

void Particle_source::particle_source_y_bottom_ge_zero( Config *conf )
{
    check_and_exit_if_not( 
	conf->sources_config_part.at(0).particle_source_y_bottom >= 0,
	"particle_source_y_bottom < 0" );
}

void Particle_source::particle_source_y_bottom_le_particle_source_y_top( Config *conf )
{
    check_and_exit_if_not( 
	conf->sources_config_part.at(0).particle_source_y_bottom <= conf->sources_config_part.at(0).particle_source_y_top,
	"particle_source_y_bottom > particle_source_y_top" );
}

void Particle_source::particle_source_y_top_le_grid_y_size( Config *conf )
{
    check_and_exit_if_not( 
	conf->sources_config_part.at(0).particle_source_y_top <= conf->mesh_config_part.grid_y_size,
	"particle_source_y_top > grid_y_size" );
}

void Particle_source::particle_source_temperature_gt_zero( Config *conf )
{
    check_and_exit_if_not( 
	conf->sources_config_part.at(0).particle_source_temperature >= 0,
	"particle_source_temperature < 0" );
}

void Particle_source::particle_source_mass_gt_zero( Config *conf )
{
    check_and_exit_if_not( 
	conf->sources_config_part.at(0).particle_source_mass >= 0,
	"particle_source_mass < 0" );
}

void check_and_exit_if_not( const bool &should_be, const std::string &message )
{
    if( !should_be ){
	std::cout << "Error: " + message << std::endl;
	exit( EXIT_FAILURE );
    }
    return;
}
