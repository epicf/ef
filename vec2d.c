#include "vec2d.h"

Vec2d vec2d_init( double x, double y )
{
    Vec2d v = { .x = {x,y} };
    return v;
}

Vec2d vec2d_zero()
{
    Vec2d v = { .x = {0.0, 0.0} };
    return v;
}

double vec2d_x( Vec2d v )
{
    return v.x[0];
}

double vec2d_y( Vec2d v )
{
    return v.x[1];
}

Vec2d vec2d_negate( Vec2d v )
{
    return vec2d_init( -vec2d_x(v), -vec2d_y(v) );
}

Vec2d vec2d_add( Vec2d v1, Vec2d v2 )
{
    return vec2d_init( vec2d_x(v1) + vec2d_x(v2), 
		       vec2d_y(v1) + vec2d_y(v2) );
}

Vec2d vec2d_sub( Vec2d v1, Vec2d v2 )
{
    return vec2d_init( vec2d_x(v1) - vec2d_x(v2), 
		       vec2d_y(v1) - vec2d_y(v2) );
}

double vec2d_dot_product( Vec2d v1, Vec2d v2 )
{
    
    return ( vec2d_x(v1) * vec2d_x(v2) + 
	     vec2d_y(v1) * vec2d_y(v2) );
}

Vec2d vec2d_times_scalar( Vec2d v, double a )
{
    return vec2d_init( a * vec2d_x(v), 
		       a * vec2d_y(v) );    
}

void vec2d_print( Vec2d v )
{
    printf("(%.2f, %.2f)", vec2d_x(v), vec2d_y(v) );
}
