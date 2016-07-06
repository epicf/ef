#include "vec3d.h"

Vec3d vec3d_init( double x, double y, double z )
{
    Vec3d v = { .x = {x,y,z} };
    return v;
}

Vec3d vec3d_zero()
{
    Vec3d v = { .x = {0.0, 0.0, 0.0} };
    return v;
}

double vec3d_x( Vec3d v )
{
    return v.x[0];
}

double vec3d_y( Vec3d v )
{
    return v.x[1];
}

double vec3d_z( Vec3d v )
{
    return v.x[2];
}

double vec3d_length( Vec3d v )
{
    return sqrt( v.x[0] * v.x[0]
		 + v.x[1] * v.x[1]
		 + v.x[2] * v.x[2] );
}

Vec3d vec3d_negate( Vec3d v )
{
    return vec3d_init( -vec3d_x(v), -vec3d_y(v), -vec3d_z(v) );
}

Vec3d vec3d_add( Vec3d v1, Vec3d v2 )
{
    return vec3d_init( vec3d_x(v1) + vec3d_x(v2), 
		       vec3d_y(v1) + vec3d_y(v2),
		       vec3d_z(v1) + vec3d_z(v2) );
}

Vec3d vec3d_sub( Vec3d v1, Vec3d v2 )
{
    return vec3d_init( vec3d_x(v1) - vec3d_x(v2), 
		       vec3d_y(v1) - vec3d_y(v2),
		       vec3d_z(v1) - vec3d_z(v2) );
}

double vec3d_dot_product( Vec3d v1, Vec3d v2 )
{
    
    return ( vec3d_x(v1) * vec3d_x(v2) + 
	     vec3d_y(v1) * vec3d_y(v2) +
	     vec3d_z(v1) * vec3d_z(v2) );
}

Vec3d vec3d_cross_product( Vec3d v1, Vec3d v2 )
{
    double prod_x = vec3d_y( v1 ) * vec3d_z( v2 ) - vec3d_z( v1 ) * vec3d_y( v2 );
    double prod_y = vec3d_z( v1 ) * vec3d_x( v2 ) - vec3d_x( v1 ) * vec3d_z( v2 );
    double prod_z = vec3d_x( v1 ) * vec3d_y( v2 ) - vec3d_y( v1 ) * vec3d_x( v2 );
    
    return vec3d_init( prod_x, prod_y, prod_z );
}

Vec3d vec3d_times_scalar( Vec3d v, double a )
{
    return vec3d_init( a * vec3d_x(v), 
		       a * vec3d_y(v),
		       a * vec3d_z(v) );    
}

Vec3d vec3d_normalized( Vec3d v )
{
    Vec3d unit_v;
    double len_v = vec3d_length( v );
    if ( len_v != 0.0 ){
	unit_v = vec3d_times_scalar( v, 1.0 / len_v );
    } else {
	unit_v = vec3d_zero();
    }
    return unit_v;
}

void vec3d_print( Vec3d v )
{
    printf("(%e, %e, %e)",
	   vec3d_x(v), vec3d_y(v), vec3d_z(v) );
}
