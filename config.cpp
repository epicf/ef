#include "config.h"

Config::Config( const std::string &filename )
{    
    try {
	using boost::property_tree::ptree;
	ptree pt;

	read_ini(filename, pt);
	
	for( auto &sections : pt ) {
	    std::string section_name = sections.first.data();
	    if ( section_name.find( "Time grid" ) != std::string::npos ) {
		time_config_part = Time_config_part( sections.second );
	    } else if ( section_name.find( "Spatial mesh" ) != std::string::npos ) {
		mesh_config_part = Mesh_config_part( sections.second );
	    } else if ( section_name.find( "Particle_source_box." ) != std::string::npos ) {
		std::string source_name = section_name.substr( section_name.find(".") + 1 );
		sources_config_part.push_back(
		    new Particle_source_box_config_part( source_name, sections.second ) );
	    } else if ( section_name.find( "Particle_source_cylinder." ) != std::string::npos ) {
		std::string source_name = section_name.substr( section_name.find(".") + 1 );
		sources_config_part.push_back(
		    new Particle_source_cylinder_config_part( source_name, sections.second ) );
	    } else if ( section_name.find( "Inner_region_box." ) != std::string::npos ) {
		std::string inner_region_name = section_name.substr( section_name.find(".") + 1 );
		inner_regions_config_part.push_back(
		    new Inner_region_box_config_part( inner_region_name, sections.second ) );
	    } else if ( section_name.find( "Inner_region_sphere." ) != std::string::npos ) {
		std::string inner_region_name = section_name.substr( section_name.find(".") + 1 );
		inner_regions_config_part.push_back(
		    new Inner_region_sphere_config_part( inner_region_name, sections.second ) );
	    } else if ( section_name.find( "Inner_region_cylinder." ) != std::string::npos ) {
		std::string inner_region_name = section_name.substr( section_name.find(".") + 1 );
		inner_regions_config_part.push_back(
		    new Inner_region_cylinder_config_part( inner_region_name, sections.second ) );
	    } else if ( section_name.find( "Inner_region_tube." ) != std::string::npos ) {
		std::string inner_region_name = section_name.substr( section_name.find(".") + 1 );
		inner_regions_config_part.push_back(
		    new Inner_region_tube_config_part( inner_region_name, sections.second ) );
	    } else if ( section_name.find( "Boundary conditions" ) != std::string::npos ) {
		boundary_config_part = Boundary_config_part( sections.second );
	    } else if ( section_name.find( "External magnetic field" ) != std::string::npos ) {
		external_magnetic_field_config_part = External_magnetic_field_config_part( sections.second );
	    } else if ( section_name.find( "Particle interaction model" ) != std::string::npos ) {
		particle_interaction_model_config_part = Particle_interaction_model_config_part( sections.second );
	    } else if ( section_name.find( "Output filename" ) != std::string::npos ) {
		output_filename_config_part = Output_filename_config_part( sections.second );				
	    } else {
		std::cout << "Ignoring unknown section: " << section_name << std::endl;
	    }
	}
	return;
    }
    catch( std::exception& e ) {
        std::cerr << "error: " << e.what() << "\n";
        exit( EXIT_FAILURE);
    }
    catch( ... ) {
        std::cerr << "Exception of unknown type!\n";
    }
}
