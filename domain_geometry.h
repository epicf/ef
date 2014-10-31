#ifndef _DOMAIN_GEOMETRY_H_
#define _DOMAIN_GEOMETRY_H_

#include <string>
#include <deal.II/base/point.h>

#include "config.h"

class Domain_geometry
{
public:
    Domain_geometry( Config &conf );
    virtual ~Domain_geometry();
    bool at_left_boundary( const dealii::Point<2> &p ) const;
    bool at_top_boundary( const dealii::Point<2> &p ) const;
    bool at_right_boundary( const dealii::Point<2> &p ) const;
    bool at_bottom_boundary( const dealii::Point<2> &p ) const;
    void write_to_file( std::ofstream &output_file );
    
public:
    double x_volume_size, y_volume_size;
    const double tolerance = 0.001;
    std::string geomerty_type = "hyper_rectangle";
    dealii::Point<2> bottom_left;
    dealii::Point<2> top_right;

private:
    void check_correctness_of_related_config_fields( Config &conf );
    void grid_x_size_gt_zero( Config &conf );
    void grid_y_size_gt_zero( Config &conf );
    void check_and_exit_if_not( const bool &should_be, const std::string &message );
};

#endif /* _DOMAIN_GEOMETRY_H_ */
