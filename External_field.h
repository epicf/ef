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
#include "physical_constants.h"
#include "lib/tinyexpr/tinyexpr.h"

class External_field{
public:
    std::string name;
    std::string field_type;
public:
    External_field( External_field_config_part &field_conf );
    External_field( hid_t h5_external_field_group_id );
    virtual Vec3d field_at_particle_position( const Particle &p, const double &t ) = 0;
    //virtual Vec3d force_on_particle( const Particle &p, const double &t ) = 0;
    void write_to_file( hid_t hdf5_file_id );
    virtual void print() {};
    virtual ~External_field() {};
protected:
    virtual void write_hdf5_field_parameters( hid_t current_field_group_id ) {};
    void hdf5_status_check( herr_t status );
};


class External_magnetic_field_uniform : public External_field
{
private:
    Vec3d magnetic_field;
public:    
    External_magnetic_field_uniform(
	External_magnetic_field_uniform_config_part &field_conf );
    External_magnetic_field_uniform( hid_t h5_external_magnetic_field_uniform_group );
    Vec3d field_at_particle_position( const Particle &p, const double &t );
    //Vec3d force_on_particle( const Particle &p, const double &t );
    virtual ~External_magnetic_field_uniform() {};
private:
    void check_correctness_of_related_config_fields(
	External_magnetic_field_uniform_config_part &field_conf );
    void get_values_from_config(
	External_magnetic_field_uniform_config_part &field_conf );
    void write_hdf5_field_parameters( hid_t current_field_group_id );
    void hdf5_status_check( herr_t status );
};



class External_electric_field_uniform : public External_field
{
private:
    Vec3d electric_field;
public:    
    External_electric_field_uniform(
	External_electric_field_uniform_config_part &field_conf );
    External_electric_field_uniform( hid_t h5_external_electric_field_uniform_group );
    Vec3d field_at_particle_position( const Particle &p, const double &t );
    //Vec3d force_on_particle( const Particle &p, const double &t );
    virtual ~External_electric_field_uniform() {};
private:
    void check_correctness_of_related_config_fields(
	External_electric_field_uniform_config_part &field_conf );
    void get_values_from_config(
	External_electric_field_uniform_config_part &field_conf );
    void write_hdf5_field_parameters( hid_t current_field_group_id );
    void hdf5_status_check( herr_t status );
};



class External_magnetic_field_tinyexpr : public External_field
{
private:
    std::string Hx_expr, Hy_expr, Hz_expr;
    te_expr *Hx, *Hy, *Hz;

    double te_x, te_y, te_z, te_t;
    //possible or not?
    //te_variable vars[] = {{"x", &te_x}, {"y", &te_y}, {"z", &te_z}, {"t", &te_t}};    
    
public:    
    External_magnetic_field_tinyexpr(
	External_magnetic_field_tinyexpr_config_part &field_conf );
    External_magnetic_field_tinyexpr( hid_t h5_external_magnetic_field_tinyexpr_group );
    Vec3d field_at_particle_position( const Particle &p, const double &t );
    //Vec3d force_on_particle( const Particle &p, const double &t );
    virtual ~External_magnetic_field_tinyexpr() {
	te_free( Hx ); te_free( Hy ); te_free( Hz );
    };
private:
    void check_correctness_and_get_values_from_config(
	External_magnetic_field_tinyexpr_config_part &field_conf );
    void write_hdf5_field_parameters( hid_t current_field_group_id );
    void hdf5_status_check( herr_t status );
};


class External_electric_field_tinyexpr : public External_field
{
private:
    std::string Ex_expr, Ey_expr, Ez_expr;
    te_expr *Ex, *Ey, *Ez;
    double te_x, te_y, te_z, te_t;
public:    
    External_electric_field_tinyexpr(
	External_electric_field_tinyexpr_config_part &field_conf );
    External_electric_field_tinyexpr( hid_t h5_external_electric_field_tinyexpr_group );
    Vec3d field_at_particle_position( const Particle &p, const double &t );
    //Vec3d force_on_particle( const Particle &p, const double &t );
    virtual ~External_electric_field_tinyexpr() {
	te_free( Ex ); te_free( Ey ); te_free( Ez );
    };
private:
    void check_correctness_and_get_values_from_config(
	External_electric_field_tinyexpr_config_part &field_conf );
    void write_hdf5_field_parameters( hid_t current_field_group_id );
    void hdf5_status_check( herr_t status );
};


class External_fields_manager{
public:
    boost::ptr_vector<External_field> electric;
    boost::ptr_vector<External_field> magnetic;
public:
    External_fields_manager( Config &conf )
    {
	for( auto &field_conf : conf.fields_config_part ){
	    if( External_magnetic_field_uniform_config_part *uni_mgn_conf =
		dynamic_cast<External_magnetic_field_uniform_config_part*>( &field_conf )){
		magnetic.push_back( new External_magnetic_field_uniform( *uni_mgn_conf ) );
	    } else if (
		External_electric_field_uniform_config_part *uni_el_conf =
		dynamic_cast<External_electric_field_uniform_config_part*>(&field_conf)){
		electric.push_back( new External_electric_field_uniform( *uni_el_conf ) );
	    } else if (
		External_magnetic_field_tinyexpr_config_part *tinyexpr_mgn_conf =
		dynamic_cast<External_magnetic_field_tinyexpr_config_part*>(&field_conf)){
		magnetic.push_back(
		    new External_magnetic_field_tinyexpr( *tinyexpr_mgn_conf ) );
	    } else if (
		External_electric_field_tinyexpr_config_part *tinyexpr_el_conf =
		dynamic_cast<External_electric_field_tinyexpr_config_part*>(&field_conf)){
		electric.push_back(
		    new External_electric_field_tinyexpr( *tinyexpr_el_conf ) );
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
	if( field_type == "magnetic_uniform" ){
	    magnetic.push_back(
		new External_magnetic_field_uniform( current_field_grpid ));
	} else if( field_type == "electric_uniform" ){
	    electric.push_back(
		new External_electric_field_uniform( current_field_grpid ));
	} else if( field_type == "magnetic_tinyexpr" ){
	    magnetic.push_back(
		new External_magnetic_field_tinyexpr( current_field_grpid ));
	} else if( field_type == "electric_tinyexpr" ){
	    electric.push_back(
		new External_electric_field_tinyexpr( current_field_grpid ));
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
	int n_of_electric_fields = electric.size();
	int n_of_magnetic_fields = magnetic.size();
	group_id = H5Gcreate2( hdf5_file_id, hdf5_groupname.c_str(),
			       H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	hdf5_status_check( group_id );

	status = H5LTset_attribute_int( hdf5_file_id,
					hdf5_groupname.c_str(),
					"number_of_electric_fields",
					&n_of_electric_fields,
					single_element );
	hdf5_status_check( status );
	status = H5LTset_attribute_int( hdf5_file_id,
					hdf5_groupname.c_str(),
					"number_of_magnetic_fields",
					&n_of_magnetic_fields,
					single_element );
	hdf5_status_check( status );
	
	for( auto &el_field : electric )
	    el_field.write_to_file( group_id );
	for( auto &mgn_field : magnetic )
	    mgn_field.write_to_file( group_id );

	status = H5Gclose( group_id );
	hdf5_status_check( status );
    }; 

    void print_fields()
    {
	for( auto &el_field : electric )
	    el_field.print();
	for( auto &mgn_field : magnetic )
	    mgn_field.print();

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
