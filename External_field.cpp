#include "External_field.h"

External_field::External_field( External_field_config_part &field_conf )
{
    name = field_conf.name;
}

External_field::External_field( hid_t h5_external_field_group )
{
    size_t grp_name_size = 0;
    char *grp_name = NULL;
    grp_name_size = H5Iget_name( h5_external_field_group, grp_name, grp_name_size );
    grp_name_size = grp_name_size + 1;
    grp_name = new char[ grp_name_size ];
    grp_name_size = H5Iget_name( h5_external_field_group, grp_name, grp_name_size );
    std::string longname = std::string( grp_name );
    name = longname.substr( longname.find_last_of("/") + 1 );
    delete[] grp_name;
}

void External_field::write_to_file( hid_t fields_group_id )
{
    hid_t current_field_group_id;
    herr_t status;
    std::string hdf5_groupname = "./" + name;
    std::string current_group = "./";

    current_field_group_id = H5Gcreate( fields_group_id, hdf5_groupname.c_str(),
					H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hdf5_status_check( current_field_group_id );


    write_hdf5_field_parameters( current_field_group_id );

    
    status = H5Gclose( current_field_group_id ); hdf5_status_check( status );
    return;
}

void External_field::hdf5_status_check( herr_t status )
{
    if( status < 0 ){
	std::cout << "Something went wrong while reading or writing External_field "
		  << name << "."
		  << "Aborting." << std::endl;
	exit( EXIT_FAILURE );
    }
}


// Uniform magnetic

External_magnetic_field_uniform::External_magnetic_field_uniform(
    External_magnetic_field_uniform_config_part &field_conf ) :
    External_field( field_conf )
{
    field_type = "magnetic_uniform";
    check_correctness_of_related_config_fields( field_conf );
    get_values_from_config( field_conf );
}

void External_magnetic_field_uniform::check_correctness_of_related_config_fields(
    External_magnetic_field_uniform_config_part &field_conf )
{
    // nothing to check here
}

void External_magnetic_field_uniform::get_values_from_config(
    External_magnetic_field_uniform_config_part &field_conf )
{
    magnetic_field = vec3d_init( field_conf.magnetic_field_x,
				 field_conf.magnetic_field_y,
				 field_conf.magnetic_field_z );
}

External_magnetic_field_uniform::External_magnetic_field_uniform(
    hid_t h5_external_magnetic_field_uniform_group ) :
    External_field( h5_external_magnetic_field_uniform_group )
{
    herr_t status;
    double H_x, H_y, H_z;

    field_type = "magnetic_uniform";
    
    status = H5LTget_attribute_double( h5_external_magnetic_field_uniform_group, "./",
				       "magnetic_uniform_field_x", &H_x );
    hdf5_status_check( status );
    status = H5LTget_attribute_double( h5_external_magnetic_field_uniform_group, "./",
				       "magnetic_uniform_field_y", &H_y );
    hdf5_status_check( status );
    status = H5LTget_attribute_double( h5_external_magnetic_field_uniform_group, "./",
				       "magnetic_uniform_field_z", &H_z );
    hdf5_status_check( status );

    magnetic_field = vec3d_init( H_x, H_y, H_z );
}

Vec3d External_magnetic_field_uniform::field_at_particle_position(
    const Particle &p, const double &t )
{
    return magnetic_field;
}

// Vec3d External_magnetic_field_uniform::force_on_particle( const Particle &p,
// 							  const double &t )
// {
//     double scale = p.charge / p.mass / speed_of_light;    
//     return vec3d_times_scalar( vec3d_cross_product( p.momentum, magnetic_field ),
// 			       scale );
// }

void External_magnetic_field_uniform::write_hdf5_field_parameters(
    hid_t current_field_group_id )
{
    double H_x = vec3d_x( magnetic_field );
    double H_y = vec3d_y( magnetic_field );
    double H_z = vec3d_z( magnetic_field );

    herr_t status;
    int single_element = 1;
    std::string current_group = "./";

    status = H5LTset_attribute_string( current_field_group_id, current_group.c_str(),
				       "field_type", field_type.c_str() );
    hdf5_status_check( status );
    status = H5LTset_attribute_double( current_field_group_id, current_group.c_str(),
				       "magnetic_uniform_field_x", &H_x, single_element );
    hdf5_status_check( status );
    status = H5LTset_attribute_double( current_field_group_id, current_group.c_str(),
				       "magnetic_uniform_field_y", &H_y, single_element );
    hdf5_status_check( status );
    status = H5LTset_attribute_double( current_field_group_id, current_group.c_str(),
				       "magnetic_uniform_field_z", &H_z, single_element );
    hdf5_status_check( status );
    status = H5LTset_attribute_double( current_field_group_id, current_group.c_str(),
				       "speed_of_light", &physconst_speed_of_light,
				       single_element );
    hdf5_status_check( status );
    
    return;
}

void External_magnetic_field_uniform::hdf5_status_check( herr_t status )
{
    if( status < 0 ){
	std::cout << "Something went wrong while reading or writing "
		  << "External_magnetic_field_uniform group. "
		  << "Aborting." << std::endl;
	exit( EXIT_FAILURE );
    }
}



// Uniform electric

External_electric_field_uniform::External_electric_field_uniform(
    External_electric_field_uniform_config_part &field_conf ) :
    External_field( field_conf )
{
    field_type = "electric_uniform";
    check_correctness_of_related_config_fields( field_conf );
    get_values_from_config( field_conf );
}

void External_electric_field_uniform::check_correctness_of_related_config_fields(
    External_electric_field_uniform_config_part &field_conf )
{
    // nothing to check here
}

void External_electric_field_uniform::get_values_from_config(
    External_electric_field_uniform_config_part &field_conf )
{
    electric_field = vec3d_init( field_conf.electric_field_x,
				 field_conf.electric_field_y,
				 field_conf.electric_field_z );
}

External_electric_field_uniform::External_electric_field_uniform(
    hid_t h5_external_electric_field_uniform_group ) :
    External_field( h5_external_electric_field_uniform_group )
{
    herr_t status;
    double E_x, E_y, E_z;

    field_type = "electric_uniform";
    
    status = H5LTget_attribute_double( h5_external_electric_field_uniform_group, "./",
				       "electric_uniform_field_x", &E_x );
    hdf5_status_check( status );
    status = H5LTget_attribute_double( h5_external_electric_field_uniform_group, "./",
				       "electric_uniform_field_y", &E_y );
    hdf5_status_check( status );
    status = H5LTget_attribute_double( h5_external_electric_field_uniform_group, "./",
				       "electric_uniform_field_z", &E_z );
    hdf5_status_check( status );

    electric_field = vec3d_init( E_x, E_y, E_z );
}

Vec3d External_electric_field_uniform::field_at_particle_position(
    const Particle &p, const double &t )
{
    return electric_field;
}

// Vec3d External_electric_field_uniform::force_on_particle( const Particle &p,
// 							  const double &t )
// {
//     double scale = p.charge / p.mass;    
//     return vec3d_times_scalar( electric_field, scale );
// }

void External_electric_field_uniform::write_hdf5_field_parameters(
    hid_t current_field_group_id )
{
    double E_x = vec3d_x( electric_field );
    double E_y = vec3d_y( electric_field );
    double E_z = vec3d_z( electric_field );

    herr_t status;
    int single_element = 1;
    std::string current_group = "./";

    status = H5LTset_attribute_string( current_field_group_id, current_group.c_str(),
				       "field_type", field_type.c_str() );
    hdf5_status_check( status );
    status = H5LTset_attribute_double( current_field_group_id, current_group.c_str(),
				       "electric_uniform_field_x", &E_x, single_element );
    hdf5_status_check( status );
    status = H5LTset_attribute_double( current_field_group_id, current_group.c_str(),
				       "electric_uniform_field_y", &E_y, single_element );
    hdf5_status_check( status );
    status = H5LTset_attribute_double( current_field_group_id, current_group.c_str(),
				       "electric_uniform_field_z", &E_z, single_element );
    hdf5_status_check( status );
    
    return;
}

void External_electric_field_uniform::hdf5_status_check( herr_t status )
{
    if( status < 0 ){
	std::cout << "Something went wrong while reading or writing "
		  << "External_electric_field_uniform group. "
		  << "Aborting." << std::endl;
	exit( EXIT_FAILURE );
    }
}



// Magnetic tinyexpr

External_magnetic_field_tinyexpr::External_magnetic_field_tinyexpr(
    External_magnetic_field_tinyexpr_config_part &field_conf ) :
    External_field( field_conf )
{
    field_type = "magnetic_tinyexpr";
    check_correctness_and_get_values_from_config( field_conf );
}

void External_magnetic_field_tinyexpr::check_correctness_and_get_values_from_config(
    External_magnetic_field_tinyexpr_config_part &field_conf )
{
    int err;
    te_variable vars[] = { {"x", &te_x}, {"y", &te_y},
			   {"z", &te_z}, {"t", &te_t} };

    Hx_expr = field_conf.magnetic_field_x;
    Hy_expr = field_conf.magnetic_field_y;
    Hz_expr = field_conf.magnetic_field_z;
    
    Hx = te_compile( Hx_expr.c_str(), vars, 4, &err );
    if ( !Hx ) {
	printf("In %s in Hx expression:\n\t%s\n", name.c_str(), Hx_expr.c_str() );
        printf("\t%*s^\nError near here\n", err-1, "");
	printf("Aboring.\n");
	exit( EXIT_FAILURE );
    }

    Hy = te_compile( Hy_expr.c_str(), vars, 4, &err );
    if ( !Hy ) {
	printf("In %s in Hy expression:\n\t%s\n", name.c_str(), Hy_expr.c_str() );
        printf("\t%*s^\nError near here\n", err-1, "");
	printf("Aboring.\n");
	exit( EXIT_FAILURE );
    }

    Hz = te_compile( Hz_expr.c_str(), vars, 4, &err );
    if ( !Hz ) {
	printf("In %s in Hz expression:\n\t%s\n", name.c_str(), Hz_expr.c_str() );
        printf("\t%*s^\nError near here\n", err-1, "");
	printf("Aboring.\n");
	exit( EXIT_FAILURE );
    }
}

External_magnetic_field_tinyexpr::External_magnetic_field_tinyexpr(
    hid_t h5_external_magnetic_field_tinyexpr_group ) :
    External_field( h5_external_magnetic_field_tinyexpr_group )
{
    herr_t status;
    char h5_str_read_buffer[1000]; // expr is supposed to be tiny

    field_type = "magnetic_tinyexpr";
    
    status = H5LTget_attribute_string( h5_external_magnetic_field_tinyexpr_group, "./",
				       "magnetic_tinyexpr_field_x",
				       h5_str_read_buffer );
    hdf5_status_check( status );
    Hx_expr = std::string( h5_str_read_buffer );
    if ( Hx_expr.length() >= 900 ) {
	printf( "Implement support for longer strings! Aborting.\n" );
	exit( EXIT_FAILURE );
    }
    
    status = H5LTget_attribute_string( h5_external_magnetic_field_tinyexpr_group, "./",
				       "magnetic_tinyexpr_field_y",
				       h5_str_read_buffer );
    hdf5_status_check( status );
    Hy_expr = std::string( h5_str_read_buffer );
    if ( Hy_expr.length() >= 900 ) {
	printf( "Implement support for longer strings! Aborting.\n" );
	exit( EXIT_FAILURE );
    }

    
    status = H5LTget_attribute_string( h5_external_magnetic_field_tinyexpr_group, "./",
				       "magnetic_tinyexpr_field_z",
				       h5_str_read_buffer );
    hdf5_status_check( status );
    Hz_expr = std::string( h5_str_read_buffer );
    if ( Hz_expr.length() >= 900 ) {
	printf( "Implement support for longer strings! Aborting.\n" );
	exit( EXIT_FAILURE );
    }

    int err;
    te_variable vars[] = { {"x", &te_x}, {"y", &te_y},
			   {"z", &te_z}, {"t", &te_t} };
    Hx = te_compile( Hx_expr.c_str(), vars, 4, &err );
    Hy = te_compile( Hy_expr.c_str(), vars, 4, &err );
    Hz = te_compile( Hz_expr.c_str(), vars, 4, &err );    
}

Vec3d External_magnetic_field_tinyexpr::field_at_particle_position(
    const Particle &p, const double &t )
{
    Vec3d pos = p.position;
    te_x = vec3d_x( pos );
    te_y = vec3d_y( pos );
    te_z = vec3d_z( pos );
    te_t = t;

    Vec3d magnetic_field = vec3d_init( te_eval( Hx ),
				       te_eval( Hy ),
				       te_eval( Hz ) );
    
    return magnetic_field;
}


// Vec3d External_magnetic_field_tinyexpr::force_on_particle( const Particle &p,
// 							   const double &t )
// {
//     double scale = p.charge / p.mass / speed_of_light;

//     Vec3d pos = p.position;
//     te_x = vec3d_x( pos );
//     te_y = vec3d_y( pos );
//     te_z = vec3d_z( pos );
//     te_t = t;

//     Vec3d magnetic_field = vec3d_init( te_eval( Hx ),
// 				       te_eval( Hy ),
// 				       te_eval( Hz ) );
    
//     return vec3d_times_scalar( vec3d_cross_product( p.momentum, magnetic_field ),
// 			       scale );
// }

void External_magnetic_field_tinyexpr::write_hdf5_field_parameters(
    hid_t current_field_group_id )
{
    herr_t status;
    int single_element = 1;
    std::string current_group = "./";

    status = H5LTset_attribute_string( current_field_group_id, current_group.c_str(),
				       "field_type",
				       field_type.c_str() );
    hdf5_status_check( status );
    status = H5LTset_attribute_string( current_field_group_id, current_group.c_str(),
				       "magnetic_tinyexpr_field_x",
				       Hx_expr.c_str() );
    hdf5_status_check( status );
    status = H5LTset_attribute_string( current_field_group_id, current_group.c_str(),
				       "magnetic_tinyexpr_field_y",
				       Hy_expr.c_str() );
    hdf5_status_check( status );
    status = H5LTset_attribute_string( current_field_group_id, current_group.c_str(),
				       "magnetic_tinyexpr_field_z",
				       Hz_expr.c_str() );
    hdf5_status_check( status );
    status = H5LTset_attribute_double( current_field_group_id, current_group.c_str(),
				       "speed_of_light", &physconst_speed_of_light,
				       single_element );
    hdf5_status_check( status );
    
    return;
}

void External_magnetic_field_tinyexpr::hdf5_status_check( herr_t status )
{
    if( status < 0 ){
	std::cout << "Something went wrong while reading or writing "
		  << "External_magnetic_field_tinyexpr group. "
		  << "Aborting." << std::endl;
	exit( EXIT_FAILURE );
    }
}



// Electric tinyexpr

External_electric_field_tinyexpr::External_electric_field_tinyexpr(
    External_electric_field_tinyexpr_config_part &field_conf ) :
    External_field( field_conf )
{
    field_type = "electric_tinyexpr";
    check_correctness_and_get_values_from_config( field_conf );
}

void External_electric_field_tinyexpr::check_correctness_and_get_values_from_config(
    External_electric_field_tinyexpr_config_part &field_conf )
{
    int err;
    te_variable vars[] = { {"x", &te_x}, {"y", &te_y},
			   {"z", &te_z}, {"t", &te_t} };

    Ex_expr = field_conf.electric_field_x;
    Ey_expr = field_conf.electric_field_y;
    Ez_expr = field_conf.electric_field_z;
    
    Ex = te_compile( Ex_expr.c_str(), vars, 4, &err );
    if ( !Ex ) {
	printf("In %s in Ex expression:\n\t%s\n", name.c_str(), Ex_expr.c_str() );
        printf("\t%*s^\nError near here\n", err-1, "");
	printf("Aboring.\n");
	exit( EXIT_FAILURE );
    }

    Ey = te_compile( Ey_expr.c_str(), vars, 4, &err );
    if ( !Ey ) {
	printf("In %s in Ey expression:\n\t%s\n", name.c_str(), Ey_expr.c_str() );
        printf("\t%*s^\nError near here\n", err-1, "");
	printf("Aboring.\n");
	exit( EXIT_FAILURE );
    }

    Ez = te_compile( Ez_expr.c_str(), vars, 4, &err );
    if ( !Ez ) {
	printf("In %s in Ez expression:\n\t%s\n", name.c_str(), Ez_expr.c_str() );
        printf("\t%*s^\nError near here\n", err-1, "");
	printf("Aboring.\n");
	exit( EXIT_FAILURE );
    }
}

External_electric_field_tinyexpr::External_electric_field_tinyexpr(
    hid_t h5_external_electric_field_tinyexpr_group ) :
    External_field( h5_external_electric_field_tinyexpr_group )
{
    herr_t status;
    char h5_str_read_buffer[1000]; // expr is supposed to be tiny

    field_type = "electric_tinyexpr";
    
    status = H5LTget_attribute_string( h5_external_electric_field_tinyexpr_group, "./",
				       "electric_tinyexpr_field_x",
				       h5_str_read_buffer );
    hdf5_status_check( status );
    Ex_expr = std::string( h5_str_read_buffer );
    if ( Ex_expr.length() >= 900 ) {
	printf( "Implement support for longer strings! Aborting.\n" );
	exit( EXIT_FAILURE );
    }
    
    status = H5LTget_attribute_string( h5_external_electric_field_tinyexpr_group, "./",
				       "electric_tinyexpr_field_y",
				       h5_str_read_buffer );
    hdf5_status_check( status );
    Ey_expr = std::string( h5_str_read_buffer );
    if ( Ey_expr.length() >= 900 ) {
	printf( "Implement support for longer strings! Aborting.\n" );
	exit( EXIT_FAILURE );
    }

    
    status = H5LTget_attribute_string( h5_external_electric_field_tinyexpr_group, "./",
				       "electric_tinyexpr_field_z",
				       h5_str_read_buffer );
    hdf5_status_check( status );
    Ez_expr = std::string( h5_str_read_buffer );
    if ( Ez_expr.length() >= 900 ) {
	printf( "Implement support for longer strings! Aborting.\n" );
	exit( EXIT_FAILURE );
    }

    int err;
    te_variable vars[] = { {"x", &te_x}, {"y", &te_y},
			   {"z", &te_z}, {"t", &te_t} };
    Ex = te_compile( Ex_expr.c_str(), vars, 4, &err );
    Ey = te_compile( Ey_expr.c_str(), vars, 4, &err );
    Ez = te_compile( Ez_expr.c_str(), vars, 4, &err );    
}

Vec3d External_electric_field_tinyexpr::field_at_particle_position(
    const Particle &p, const double &t )
{
    Vec3d pos = p.position;
    te_x = vec3d_x( pos );
    te_y = vec3d_y( pos );
    te_z = vec3d_z( pos );
    te_t = t;

    Vec3d electric_field = vec3d_init( te_eval( Ex ),
				       te_eval( Ey ),
				       te_eval( Ez ) );
    
    return electric_field;
}

// Vec3d External_electric_field_tinyexpr::force_on_particle( const Particle &p,
// 							   const double &t )
// {
//     double scale = p.charge / p.mass;

//     Vec3d pos = p.position;
//     te_x = vec3d_x( pos );
//     te_y = vec3d_y( pos );
//     te_z = vec3d_z( pos );
//     te_t = t;

//     Vec3d electric_field = vec3d_init( te_eval( Ex ),
// 				       te_eval( Ey ),
// 				       te_eval( Ez ) );
    
//     return vec3d_times_scalar( electric_field, scale );
// }

void External_electric_field_tinyexpr::write_hdf5_field_parameters(
    hid_t current_field_group_id )
{
    herr_t status;
    std::string current_group = "./";

    status = H5LTset_attribute_string( current_field_group_id, current_group.c_str(),
				       "field_type",
				       field_type.c_str() );
    hdf5_status_check( status );
    status = H5LTset_attribute_string( current_field_group_id, current_group.c_str(),
				       "electric_tinyexpr_field_x",
				       Ex_expr.c_str() );
    hdf5_status_check( status );
    status = H5LTset_attribute_string( current_field_group_id, current_group.c_str(),
				       "electric_tinyexpr_field_y",
				       Ey_expr.c_str() );
    hdf5_status_check( status );
    status = H5LTset_attribute_string( current_field_group_id, current_group.c_str(),
				       "electric_tinyexpr_field_z",
				       Ez_expr.c_str() );
    hdf5_status_check( status );
    
    return;
}

void External_electric_field_tinyexpr::hdf5_status_check( herr_t status )
{
    if( status < 0 ){
	std::cout << "Something went wrong while reading or writing "
		  << "External_electric_field_tinyexpr group. "
		  << "Aborting." << std::endl;
	exit( EXIT_FAILURE );
    }
}




// RectGrid magnetic

External_magnetic_field_RectGrid::External_magnetic_field_RectGrid(
    External_magnetic_field_RectGrid_config_part &field_conf ) :
    External_field( field_conf )
{
    field_type = "magnetic_rectgrid";
    check_correctness_of_related_config_fields( field_conf );
    get_values_from_config( field_conf );
    init_fields_from_file();
}

void External_magnetic_field_RectGrid::check_correctness_of_related_config_fields(
    External_magnetic_field_RectGrid_config_part &field_conf )
{
    std::string test_filename = field_conf.field_filename;
    std::ifstream f( test_filename.c_str() );
    if ( f.good() ){
	printf("good field file: %s", test_filename.c_str() );
    } else {
	printf("bad field file: %s", test_filename.c_str() );
	printf("Aboring.\n");
	exit( EXIT_FAILURE );
    }
}

void External_magnetic_field_RectGrid::get_values_from_config(
    External_magnetic_field_RectGrid_config_part &field_conf )
{
    field_filename = field_conf.field_filename;
}

External_magnetic_field_RectGrid::External_magnetic_field_RectGrid(
    hid_t h5_external_magnetic_field_RectGrid_group ) :
    External_field( h5_external_magnetic_field_RectGrid_group )
{
    herr_t status;
    field_type = "magnetic_rectgrid";
    
    char h5_str_read_buffer[200]; // todo: accept longer filenames
    
    status = H5LTget_attribute_string( h5_external_magnetic_field_RectGrid_group, "./",
				       "field_filename",
				       h5_str_read_buffer );
    hdf5_status_check( status );
    field_filename = std::string( h5_str_read_buffer );

    
    // todo: split into functions
    std::ifstream f( field_filename.c_str() );
    if ( f.good() ){
	printf("good field file: %s", field_filename.c_str() );
    } else {
	printf("bad field file: %s", field_filename.c_str() );
	printf("Aboring.\n");
	exit( EXIT_FAILURE );
    }
    f.close();

    // 
    init_fields_from_file();
}

void External_magnetic_field_RectGrid::init_fields_from_file()
{
    std::ifstream f( field_filename.c_str() );
    std::string line;
    std::string::size_type sz;
    std::stringstream ss; // cpp trick to read string word by word
    std::string word;
    
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> z;
    std::vector<double> Bx;
    std::vector<double> By;
    std::vector<double> Bz;
    std::vector<double> Ex;
    std::vector<double> Ey;
    std::vector<double> Ez;
    
    while( getline( f, line ) ){
	ss.str( line );
	ss >> word;
	x.push_back( std::stod( word ) );
	//y.push_back( std::stod( line.substr( sz ), &sz ) ); doesn't work
	//z.push_back( std::stod( line.substr( sz ), &sz ) ); doesn't work
	ss >> word;
	y.push_back( std::stod( word ) );
	ss >> word;
	z.push_back( std::stod( word ) );
	ss >> word;
	Bx.push_back( std::stod( word ) );
	ss >> word;
	By.push_back( std::stod( word ) );
	ss >> word;
	Bz.push_back( std::stod( word ) );
	ss >> word;
	Ex.push_back( std::stod( word ) );
	ss >> word;
	Ey.push_back( std::stod( word ) );
	ss >> word;
	Ez.push_back( std::stod( word ) );
	// printf( "xyz: %f, %f, %f \n", x.back(),  y.back(),  z.back() );
	// printf( "B: %f, %f, %f \n", Bx.back(), By.back(), Bz.back() );
	// printf( "E: %f, %f, %f \n", Ex.back(), Ey.back(), Ez.back() );
    }
    f.close();
    
    std::set<double> x_nodes( x.begin(), x.end() );    
    x_start = *( x_nodes.begin() ); // no [i] for sets
    x_end = *( x_nodes.rbegin() ); // x_nodes.end() - 1 doesn't work
    dx = *( std::next( x_nodes.begin() ) ) - *( x_nodes.begin() );
    x_n_nodes = x_nodes.size();
    // printf( "xset: %f, %f, %f, %d \n", x_start, x_end, dx, x_n_nodes );
    // for (int i = 0; i < x_nodes.size(); ++i){
    // 	printf( "%f ", *( std::next( x_nodes.begin(), i ) ) ); 
    // }
    std::set<double> y_nodes( y.begin(), y.end() );
    y_start = *( y_nodes.begin() );
    y_end = *( y_nodes.rbegin() );
    dy = *( std::next( y_nodes.begin() ) ) - *( y_nodes.begin() );
    y_n_nodes = y_nodes.size();
    // printf( "%f, %f, %f, %d \n", y_start, y_end, dy, y_n_nodes );
    std::set<double> z_nodes( z.begin(), z.end() );
    z_start = *( z_nodes.begin() );
    z_end = *( z_nodes.rbegin() );
    dz = *( std::next( z_nodes.begin() ) ) - *( z_nodes.begin() );
    // for (int i = 0; i < z_nodes.size(); ++i){
    // 	printf( "%f ", *( std::next( z_nodes.begin(), i ) ) ); 
    // }    
    z_n_nodes = z_nodes.size();
    // printf( "%f, %f, %f, %d \n", z_start, z_end, dz, z_n_nodes );
    // printf( "nodes: %d, %d \ n", Bx.size(), x_n_nodes * y_n_nodes * z_n_nodes );

    magnetic_field.resize( boost::extents[x_n_nodes][y_n_nodes][z_n_nodes] );
    
    // // todo: assert order. Is it x, y, z? 
    for ( unsigned int i = 0; i < Bx.size(); i++ ){	
    	( magnetic_field.data() )[i] = vec3d_init( Bx[i], By[i], Bz[i] );
    }
}

Vec3d External_magnetic_field_RectGrid::field_at_particle_position(
    const Particle &p, const double &t )
{
    bool out = 
	( vec3d_x( p.position ) >= x_end ) || ( vec3d_x( p.position ) <= x_start ) ||
	( vec3d_y( p.position ) >= y_end ) || ( vec3d_y( p.position ) <= y_start ) ||
	( vec3d_z( p.position ) >= z_end ) || ( vec3d_z( p.position ) <= z_start ) ;
    if( out ){
	printf("particle not inside field grid \n");
	return vec3d_zero();
    }
    
    // copied from Particle_to_mesh_map::field_at_particle_position
    // todo: avoid copy. Make grid class?
    int tlf_i, tlf_j, tlf_k; // 'tlf' = 'top_left_far'
    double tlf_x_weight, tlf_y_weight, tlf_z_weight;  
    Vec3d field_from_node, total_field;
    //
    next_node_num_and_weight( vec3d_x( p.position ), dx, &tlf_i, &tlf_x_weight );
    next_node_num_and_weight( vec3d_y( p.position ), dy, &tlf_j, &tlf_y_weight );
    next_node_num_and_weight( vec3d_z( p.position ), dz, &tlf_k, &tlf_z_weight );
    // tlf
    total_field = vec3d_zero();
    field_from_node = vec3d_times_scalar(
	magnetic_field[tlf_i][tlf_j][tlf_k],
	tlf_x_weight );
    field_from_node = vec3d_times_scalar( field_from_node, tlf_y_weight );
    field_from_node = vec3d_times_scalar( field_from_node, tlf_z_weight );
    total_field = vec3d_add( total_field, field_from_node );
    // trf
    field_from_node = vec3d_times_scalar(
	magnetic_field[tlf_i-1][tlf_j][tlf_k],
	1.0 - tlf_x_weight );
    field_from_node = vec3d_times_scalar( field_from_node, tlf_y_weight );
    field_from_node = vec3d_times_scalar( field_from_node, tlf_z_weight );
    total_field = vec3d_add( total_field, field_from_node );
    // blf
    field_from_node = vec3d_times_scalar(
	magnetic_field[tlf_i][tlf_j - 1][tlf_k],	
	tlf_x_weight );
    field_from_node = vec3d_times_scalar( field_from_node, 1.0 - tlf_y_weight );
    field_from_node = vec3d_times_scalar( field_from_node, tlf_z_weight );
    total_field = vec3d_add( total_field, field_from_node );
    // brf
    field_from_node = vec3d_times_scalar(			
	magnetic_field[tlf_i-1][tlf_j-1][tlf_k],	
	1.0 - tlf_x_weight );
    field_from_node = vec3d_times_scalar( field_from_node, 1.0 - tlf_y_weight );
    field_from_node = vec3d_times_scalar( field_from_node, tlf_z_weight );
    total_field = vec3d_add( total_field, field_from_node );
    // tln
    field_from_node = vec3d_times_scalar(
	magnetic_field[tlf_i][tlf_j][tlf_k-1],
	tlf_x_weight );
    field_from_node = vec3d_times_scalar( field_from_node, tlf_y_weight );
    field_from_node = vec3d_times_scalar( field_from_node, 1.0 - tlf_z_weight );
    total_field = vec3d_add( total_field, field_from_node );
    // trn
    field_from_node = vec3d_times_scalar(
	magnetic_field[tlf_i-1][tlf_j][tlf_k-1],
	1.0 - tlf_x_weight );
    field_from_node = vec3d_times_scalar( field_from_node, tlf_y_weight );
    field_from_node = vec3d_times_scalar( field_from_node, 1.0 - tlf_z_weight );
    total_field = vec3d_add( total_field, field_from_node );
    // bln
    field_from_node = vec3d_times_scalar(
	magnetic_field[tlf_i][tlf_j - 1][tlf_k-1],	
	tlf_x_weight );
    field_from_node = vec3d_times_scalar( field_from_node, 1.0 - tlf_y_weight );
    field_from_node = vec3d_times_scalar( field_from_node, 1.0 - tlf_z_weight );
    total_field = vec3d_add( total_field, field_from_node );
    // brn
    field_from_node = vec3d_times_scalar(			
	magnetic_field[tlf_i-1][tlf_j-1][tlf_k-1],	
	1.0 - tlf_x_weight );
    field_from_node = vec3d_times_scalar( field_from_node, 1.0 - tlf_y_weight );
    field_from_node = vec3d_times_scalar( field_from_node, 1.0 - tlf_z_weight );
    total_field = vec3d_add( total_field, field_from_node );
    //vec3d_print( total_field );
    //
    return total_field;
}

void External_magnetic_field_RectGrid::next_node_num_and_weight( 
    const double x, const double grid_step, 
    int *next_node, double *weight )
{
    double x_in_grid_units = x / grid_step;
    *next_node = ceil( x_in_grid_units );
    *weight = 1.0 - ( *next_node - x_in_grid_units );
    return;
}


void External_magnetic_field_RectGrid::write_hdf5_field_parameters(
    hid_t current_field_group_id )
{
    herr_t status;
    // int single_element = 1;
    std::string current_group = "./";

    status = H5LTset_attribute_string( current_field_group_id, current_group.c_str(),
				       "field_type", field_type.c_str() );
    hdf5_status_check( status );
    status = H5LTset_attribute_string( current_field_group_id, current_group.c_str(),
				       "field_filename", field_filename.c_str() );
    hdf5_status_check( status );

    // todo: decide whether to write fields into h5 or not. 
    
    return;
}

void External_magnetic_field_RectGrid::hdf5_status_check( herr_t status )
{
    if( status < 0 ){
	std::cout << "Something went wrong while reading or writing "
		  << "External_magnetic_field_RectGrid group. "
		  << "Aborting." << std::endl;
	exit( EXIT_FAILURE );
    }
}




// RectGrid electric

External_electric_field_RectGrid::External_electric_field_RectGrid(
    External_electric_field_RectGrid_config_part &field_conf ) :
    External_field( field_conf )
{
    field_type = "electric_rectgrid";
    check_correctness_of_related_config_fields( field_conf );
    get_values_from_config( field_conf );
    init_fields_from_file();
}

void External_electric_field_RectGrid::check_correctness_of_related_config_fields(
    External_electric_field_RectGrid_config_part &field_conf )
{
    std::string test_filename = field_conf.field_filename;
    std::ifstream f( test_filename.c_str() );
    if ( f.good() ){
	printf("good field file: %s", test_filename.c_str() );
    } else {
	printf("bad field file: %s", test_filename.c_str() );
	printf("Aboring.\n");
	exit( EXIT_FAILURE );
    }
}

void External_electric_field_RectGrid::get_values_from_config(
    External_electric_field_RectGrid_config_part &field_conf )
{
    field_filename = field_conf.field_filename;
}

External_electric_field_RectGrid::External_electric_field_RectGrid(
    hid_t h5_external_electric_field_RectGrid_group ) :
    External_field( h5_external_electric_field_RectGrid_group )
{
    herr_t status;
    field_type = "electric_rectgrid";
    
    char h5_str_read_buffer[200]; // todo: accept longer filenames
    
    status = H5LTget_attribute_string( h5_external_electric_field_RectGrid_group, "./",
				       "field_filename",
				       h5_str_read_buffer );
    hdf5_status_check( status );
    field_filename = std::string( h5_str_read_buffer );

    
    // todo: split into functions
    std::ifstream f( field_filename.c_str() );
    if ( f.good() ){
	printf("good field file: %s", field_filename.c_str() );
    } else {
	printf("bad field file: %s", field_filename.c_str() );
	printf("Aboring.\n");
	exit( EXIT_FAILURE );
    }
    f.close();

    // 
    init_fields_from_file();
}

void External_electric_field_RectGrid::init_fields_from_file()
{
    std::ifstream f( field_filename.c_str() );
    std::string line;
    std::string::size_type sz;
    
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> z;
    std::vector<double> Bx;
    std::vector<double> By;
    std::vector<double> Bz;
    std::vector<double> Ex;
    std::vector<double> Ey;
    std::vector<double> Ez;
    
    while( getline( f, line ) ){
	x.push_back( std::stod( line, &sz ) );
	y.push_back( std::stod( line.substr( sz ), &sz ) );
	z.push_back( std::stod( line.substr( sz ), &sz ) );
	Bx.push_back( std::stod( line.substr( sz ), &sz ) );
	By.push_back( std::stod( line.substr( sz ), &sz ) );
	Bz.push_back( std::stod( line.substr( sz ), &sz ) );
	Ex.push_back( std::stod( line.substr( sz ), &sz ) );
	Ey.push_back( std::stod( line.substr( sz ), &sz ) );
	Ez.push_back( std::stod( line.substr( sz ), &sz ) );	
    }
    f.close();
    
    std::set<double> x_nodes( x.begin(), x.end() );    
    x_start = *( x.begin() ); // no [i] for sets
    x_end = *( x.end() );
    dx = *( std::next( x.begin() ) ) - *( x.begin() );
    x_n_nodes = x_nodes.size();
    std::set<double> y_nodes( y.begin(), y.end() );
    y_start = *( y.begin() );
    y_end = *( y.end() );
    dy = *( std::next( y.begin() ) ) - *( y.begin() );
    y_n_nodes = y_nodes.size();
    std::set<double> z_nodes( z.begin(), z.end() );
    z_start = *( z.begin() );
    z_end = *( z.end() );
    dz = *( std::next( z.begin() ) ) - *( z.begin() );
    z_n_nodes = z_nodes.size();
    
    electric_field.resize( boost::extents[x_n_nodes][y_n_nodes][z_n_nodes] );

    // todo: assert order. Is it x, y, z? 
    for ( unsigned int i = 0; i < Ex.size(); i++ ){	
	( electric_field.data() )[i] = vec3d_init( Ex[i], Ey[i], Ez[i] );
    }
}

Vec3d External_electric_field_RectGrid::field_at_particle_position(
    const Particle &p, const double &t )
{
    bool out = 
	( vec3d_x( p.position ) >= x_end ) || ( vec3d_x( p.position ) <= x_start ) ||
	( vec3d_y( p.position ) >= y_end ) || ( vec3d_y( p.position ) <= y_start ) ||
	( vec3d_z( p.position ) >= z_end ) || ( vec3d_z( p.position ) <= z_start ) ;
    if( out ){
	return vec3d_zero();
    }
    
    // copied from Particle_to_mesh_map::field_at_particle_position
    // todo: avoid copy. Make grid class?
    int tlf_i, tlf_j, tlf_k; // 'tlf' = 'top_left_far'
    double tlf_x_weight, tlf_y_weight, tlf_z_weight;  
    Vec3d field_from_node, total_field;
    //
    next_node_num_and_weight( vec3d_x( p.position ), dx, &tlf_i, &tlf_x_weight );
    next_node_num_and_weight( vec3d_y( p.position ), dy, &tlf_j, &tlf_y_weight );
    next_node_num_and_weight( vec3d_z( p.position ), dz, &tlf_k, &tlf_z_weight );
    // tlf
    total_field = vec3d_zero();
    field_from_node = vec3d_times_scalar(
	electric_field[tlf_i][tlf_j][tlf_k],
	tlf_x_weight );
    field_from_node = vec3d_times_scalar( field_from_node, tlf_y_weight );
    field_from_node = vec3d_times_scalar( field_from_node, tlf_z_weight );
    total_field = vec3d_add( total_field, field_from_node );
    // trf
    field_from_node = vec3d_times_scalar(
	electric_field[tlf_i-1][tlf_j][tlf_k],
	1.0 - tlf_x_weight );
    field_from_node = vec3d_times_scalar( field_from_node, tlf_y_weight );
    field_from_node = vec3d_times_scalar( field_from_node, tlf_z_weight );
    total_field = vec3d_add( total_field, field_from_node );
    // blf
    field_from_node = vec3d_times_scalar(
	electric_field[tlf_i][tlf_j - 1][tlf_k],	
	tlf_x_weight );
    field_from_node = vec3d_times_scalar( field_from_node, 1.0 - tlf_y_weight );
    field_from_node = vec3d_times_scalar( field_from_node, tlf_z_weight );
    total_field = vec3d_add( total_field, field_from_node );
    // brf
    field_from_node = vec3d_times_scalar(			
	electric_field[tlf_i-1][tlf_j-1][tlf_k],	
	1.0 - tlf_x_weight );
    field_from_node = vec3d_times_scalar( field_from_node, 1.0 - tlf_y_weight );
    field_from_node = vec3d_times_scalar( field_from_node, tlf_z_weight );
    total_field = vec3d_add( total_field, field_from_node );
    // tln
    field_from_node = vec3d_times_scalar(
	electric_field[tlf_i][tlf_j][tlf_k-1],
	tlf_x_weight );
    field_from_node = vec3d_times_scalar( field_from_node, tlf_y_weight );
    field_from_node = vec3d_times_scalar( field_from_node, 1.0 - tlf_z_weight );
    total_field = vec3d_add( total_field, field_from_node );
    // trn
    field_from_node = vec3d_times_scalar(
	electric_field[tlf_i-1][tlf_j][tlf_k-1],
	1.0 - tlf_x_weight );
    field_from_node = vec3d_times_scalar( field_from_node, tlf_y_weight );
    field_from_node = vec3d_times_scalar( field_from_node, 1.0 - tlf_z_weight );
    total_field = vec3d_add( total_field, field_from_node );
    // bln
    field_from_node = vec3d_times_scalar(
	electric_field[tlf_i][tlf_j - 1][tlf_k-1],	
	tlf_x_weight );
    field_from_node = vec3d_times_scalar( field_from_node, 1.0 - tlf_y_weight );
    field_from_node = vec3d_times_scalar( field_from_node, 1.0 - tlf_z_weight );
    total_field = vec3d_add( total_field, field_from_node );
    // brn
    field_from_node = vec3d_times_scalar(			
	electric_field[tlf_i-1][tlf_j-1][tlf_k-1],	
	1.0 - tlf_x_weight );
    field_from_node = vec3d_times_scalar( field_from_node, 1.0 - tlf_y_weight );
    field_from_node = vec3d_times_scalar( field_from_node, 1.0 - tlf_z_weight );
    total_field = vec3d_add( total_field, field_from_node );    
    //
    return total_field;
}

void External_electric_field_RectGrid::next_node_num_and_weight( 
    const double x, const double grid_step, 
    int *next_node, double *weight )
{
    double x_in_grid_units = x / grid_step;
    *next_node = ceil( x_in_grid_units );
    *weight = 1.0 - ( *next_node - x_in_grid_units );
    return;
}


void External_electric_field_RectGrid::write_hdf5_field_parameters(
    hid_t current_field_group_id )
{
    herr_t status;
    // int single_element = 1;
    std::string current_group = "./";

    status = H5LTset_attribute_string( current_field_group_id, current_group.c_str(),
				       "field_type", field_type.c_str() );
    hdf5_status_check( status );
    status = H5LTset_attribute_string( current_field_group_id, current_group.c_str(),
				       "field_filename", field_filename.c_str() );
    hdf5_status_check( status );

    // todo: decide whether to write fields into h5 or not. 
    
    return;
}

void External_electric_field_RectGrid::hdf5_status_check( herr_t status )
{
    if( status < 0 ){
	std::cout << "Something went wrong while reading or writing "
		  << "External_electric_field_RectGrid group. "
		  << "Aborting." << std::endl;
	exit( EXIT_FAILURE );
    }
}
