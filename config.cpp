#include "config.h"

Config::Config( const std::string &filename )
{
    try {
        using boost::property_tree::ptree;
        ptree pt;

        read_ini(filename, pt);

        for( auto &sections : pt ) {
            std::string section_name = sections.first.data();
            if ( section_name.find( "TimeGrid" ) != std::string::npos ) {
                time_config_part = Time_config_part( sections.second );
            } else if ( section_name.find( "SpatialMesh" ) != std::string::npos ) {
                mesh_config_part = Mesh_config_part( sections.second );
            } else if (
                section_name.find( "ParticleSourceBox." ) != std::string::npos ) {
                std::string source_name = section_name.substr(section_name.find(".") + 1);
                sources_config_part.push_back(
                    new Particle_source_box_config_part( source_name, sections.second ) );
            } else if (
                section_name.find( "ParticleSourceCylinder." ) != std::string::npos ) {
                std::string source_name = section_name.substr(section_name.find(".") + 1);
                sources_config_part.push_back(
                    new Particle_source_cylinder_config_part(
                        source_name, sections.second ) );
            } else if (
                section_name.find("ParticleSourceTubeAlongZ." ) != std::string::npos){
                std::string source_name = section_name.substr(section_name.find(".") + 1);
                sources_config_part.push_back(
                    new Particle_source_tube_along_z_config_part(
                        source_name, sections.second ) );
            } else if ( section_name.find( "InnerRegionBox." ) != std::string::npos ) {
                std::string inner_region_name =
                    section_name.substr( section_name.find(".") + 1 );
                inner_regions_config_part.push_back(
                    new Inner_region_box_config_part(
                        inner_region_name, sections.second ) );
            } else if ( section_name.find("InnerRegionSphere.") != std::string::npos ) {
                std::string inner_region_name =
                    section_name.substr( section_name.find(".") + 1 );
                inner_regions_config_part.push_back(
                    new Inner_region_sphere_config_part(
                        inner_region_name, sections.second ) );
            } else if (
                section_name.find( "InnerRegionCylinder." ) != std::string::npos ) {
                std::string inner_region_name =
                    section_name.substr( section_name.find(".") + 1 );
                inner_regions_config_part.push_back(
                    new Inner_region_cylinder_config_part(
                        inner_region_name, sections.second ) );
            } else if ( section_name.find( "InnerRegionTube." ) != std::string::npos ) {
                std::string inner_region_name =
                    section_name.substr( section_name.find(".") + 1 );
                inner_regions_config_part.push_back(
                    new Inner_region_tube_config_part(
                        inner_region_name, sections.second ) );
            } else if ( section_name.find( "InnerRegionTubeAlongZSegment." )
                        != std::string::npos ) {
                std::string inner_region_name =
                    section_name.substr( section_name.find(".") + 1 );
                inner_regions_config_part.push_back(
                    new Inner_region_tube_along_z_segment_config_part(
                        inner_region_name, sections.second ) );
            } else if ( section_name.find( "InnerRegionConeAlongZ." )
                        != std::string::npos ) {
                std::string inner_region_name =
                    section_name.substr( section_name.find(".") + 1 );
                inner_regions_config_part.push_back(
                    new Inner_region_cone_along_z_config_part(
                        inner_region_name, sections.second ) );
            } else if ( section_name.find( "BoundaryConditions" ) != std::string::npos ) {
                boundary_config_part = Boundary_config_part( sections.second );
            } else if ( section_name.find(
                            "ExternalMagneticFieldUniform" ) != std::string::npos ) {
                std::string field_name =
                    section_name.substr( section_name.find(".") + 1 );
                fields_config_part.push_back(
                    new External_magnetic_field_uniform_config_part(
                        field_name, sections.second ) );
            } else if ( section_name.find(
                            "ExternalElectricFieldUniform" ) != std::string::npos ) {
                std::string field_name =
                    section_name.substr( section_name.find(".") + 1 );
                fields_config_part.push_back(
                    new External_electric_field_uniform_config_part(
                        field_name, sections.second ) );
            } else if ( section_name.find(
                            "ExternalMagneticFieldTinyexpr" ) != std::string::npos ) {
                std::string field_name =
                    section_name.substr( section_name.find(".") + 1 );
                fields_config_part.push_back(
                    new External_magnetic_field_tinyexpr_config_part(
                        field_name, sections.second ) );
            } else if ( section_name.find(
                            "ExternalElectricFieldTinyexpr" ) != std::string::npos ) {
                std::string field_name =
                    section_name.substr( section_name.find(".") + 1 );
                fields_config_part.push_back(
                    new External_electric_field_tinyexpr_config_part(
                        field_name, sections.second ) );
            } else if (
                section_name.find( "ExternalElectricFieldOnRegularGrid" ) != std::string::npos
                ||
                section_name.find("ExternalFieldElectricOnRegularGridFromH5File" )
                != std::string::npos ) {
                std::string field_name =
                    section_name.substr( section_name.find(".") + 1 );
                fields_config_part.push_back(
                    new External_electric_field_on_regular_grid_config_part(
                        field_name, sections.second ) );
            } else if (
                section_name.find( "ParticleInteractionModel" ) != std::string::npos ) {
                particle_interaction_model_config_part =
                    Particle_interaction_model_config_part( sections.second );
            } else if ( section_name.find( "OutputFilename" ) != std::string::npos ) {
                output_filename_config_part =
                    Output_filename_config_part( sections.second );
            } else {
                std::cout << "Warning! Unknown section: " << section_name << std::endl;
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
