#ifndef _EXTERNAL_FIELD_H_
#define _EXTERNAL_FIELD_H_

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <algorithm>
#include <boost/ptr_container/ptr_vector.hpp>
#include <hdf5.h>
#include <hdf5_hl.h>
#include <mpi.h>
#include "config.h"
#include "particle.h"
#include "vec3d.h"
#include "lib/tinyexpr/tinyexpr.h"

class External_field{
public:
    std::string name;
    std::string field_type;
public:
    External_field( External_field_config_part &field_conf );
    External_field( hid_t h5_external_field_group_id );
    virtual Vec3d force_on_particle( const Particle &p, const double &t ) = 0;
    void write_to_file( hid_t hdf5_file_id );
    virtual void print() {};
    virtual ~External_field() {};
protected:
    virtual void write_hdf5_field_parameters( hid_t current_field_group_id ) {};
    void hdf5_status_check( herr_t status );
};


class External_field_uniform_magnetic : public External_field
{
private:
    Vec3d magnetic_field;
    double speed_of_light;
public:    
    External_field_uniform_magnetic(
	External_field_uniform_magnetic_config_part &field_conf );
    External_field_uniform_magnetic( hid_t h5_external_field_uniform_magnetic_group );
    Vec3d force_on_particle( const Particle &p, const double &t );
    virtual ~External_field_uniform_magnetic() {};
private:
    void check_correctness_of_related_config_fields(
	External_field_uniform_magnetic_config_part &field_conf );
    void get_values_from_config(
	External_field_uniform_magnetic_config_part &field_conf );
    void write_hdf5_field_parameters( hid_t current_field_group_id );
    void hdf5_status_check( herr_t status );
};



class External_field_tinyexpr_magnetic : public External_field
{
private:
    std::string Hx_expr, Hy_expr, Hz_expr;
    te_expr *Hx, *Hy, *Hz;

    double te_x, te_y, te_z, te_t;
    //possible or not?
    //te_variable vars[] = {{"x", &x}, {"y", &y}, {"z", &z}, {"t", &t}};    
    
    double speed_of_light;    
public:    
    External_field_tinyexpr_magnetic(
	External_field_tinyexpr_magnetic_config_part &field_conf );
    External_field_tinyexpr_magnetic( hid_t h5_external_field_tinyexpr_magnetic_group );
    Vec3d force_on_particle( const Particle &p, const double &t );
    virtual ~External_field_tinyexpr_magnetic() {
	te_free( Hx ); te_free( Hy ); te_free( Hz );
    };
private:
    void check_correctness_and_get_values_from_config(
	External_field_tinyexpr_magnetic_config_part &field_conf );
    void write_hdf5_field_parameters( hid_t current_field_group_id );
    void hdf5_status_check( herr_t status );
};



class External_fields_manager{
public:
    boost::ptr_vector<External_field> fields;
public:
    External_fields_manager( Config &conf )
    {
	for( auto &field_conf : conf.fields_config_part ){
	    if( External_field_uniform_magnetic_config_part *uni_mgn_conf =
		dynamic_cast<External_field_uniform_magnetic_config_part*>( &field_conf )){
		fields.push_back( new External_field_uniform_magnetic( *uni_mgn_conf ) );
	    } else if (
		External_field_tinyexpr_magnetic_config_part *tinyexpr_mgn_conf =
		dynamic_cast<External_field_tinyexpr_magnetic_config_part*>(&field_conf)){
		fields.push_back(
		    new External_field_tinyexpr_magnetic( *tinyexpr_mgn_conf ) );
	    } else {
		std::cout << "In fields_manager constructor: " 
			  << "Unknown config type. Aborting" << std::endl; 
		exit( EXIT_FAILURE );
	    }
	}
    }

    External_fields_manager( hid_t h5_external_fields_group )
    {
	hsize_t nobj;
	ssize_t len;
	herr_t err;
	int otype;
	size_t MAX_NAME = 1024;
	char memb_name_cstr[MAX_NAME];
	hid_t current_field_grpid;
	err = H5Gget_num_objs(h5_external_fields_group, &nobj);

	for( hsize_t i = 0; i < nobj; i++ ){
	    len = H5Gget_objname_by_idx( h5_external_fields_group, i, 
					 memb_name_cstr, MAX_NAME );
	    hdf5_status_check( len );
	    otype = H5Gget_objtype_by_idx( h5_external_fields_group, i );
	    if ( otype == H5G_GROUP ) {
		current_field_grpid = H5Gopen( h5_external_fields_group,
					       memb_name_cstr, H5P_DEFAULT );
		hdf5_status_check( err );

		parse_hdf5_external_field( current_field_grpid );

		err = H5Gclose( current_field_grpid ); hdf5_status_check( err );
	    }		
	}
    }

    void parse_hdf5_external_field( hid_t current_field_grpid )
    {
	herr_t status;
	char field_type_cstr[50];
	status = H5LTget_attribute_string( current_field_grpid, "./",
					   "field_type", field_type_cstr );
	hdf5_status_check( status );

	std::string field_type( field_type_cstr );
	if( field_type == "uniform_magnetic" ){
	    fields.push_back( new External_field_uniform_magnetic( current_field_grpid ));
	} else if( field_type == "tinyexpr_magnetic" ){
	    fields.push_back( new External_field_tinyexpr_magnetic( current_field_grpid ));
	} else {
	    std::cout << "In External_field_manager constructor-from-h5: "
		      << "Unknown external_field type. Aborting"
		      << std::endl;
	    exit( EXIT_FAILURE );
	}	
    }
    
    virtual ~External_fields_manager() {};

    void write_to_file( hid_t hdf5_file_id )
    {
	hid_t group_id;
	herr_t status;
	int single_element = 1;
	std::string hdf5_groupname = "/External_fields";
	int n_of_fields = fields.size();
	group_id = H5Gcreate2( hdf5_file_id, hdf5_groupname.c_str(),
			       H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	hdf5_status_check( group_id );

	status = H5LTset_attribute_int( hdf5_file_id,
					hdf5_groupname.c_str(),
					"number_of_fields", &n_of_fields,
					single_element );
	hdf5_status_check( status );
	
	for( auto &field : fields )
	    field.write_to_file( group_id );

	status = H5Gclose( group_id );
	hdf5_status_check( status );
    }; 

    void print_fields()
    {
	for( auto &field : fields )
	    field.print();
    };

    void hdf5_status_check( herr_t status )
    {
	if( status < 0 ){
	    std::cout << "Something went wrong while writing or reading"
		      << "'External_fields' group. Aborting."
		      << std::endl;
	    exit( EXIT_FAILURE );
	}
    };
};

#endif /* _EXTERNAL_FIELD_H_ */
