#ifndef _VEC_H_
#define _VEC_H_

#include <stdio.h>

typedef struct {
    double x[2];
} Vec2d;

Vec2d vec2d_init( double x, double y );
Vec2d vec2d_zero( );
double vec2d_x( Vec2d v );
double vec2d_y( Vec2d v );
Vec2d vec2d_negate( Vec2d v );
Vec2d vec2d_add( Vec2d v1, Vec2d v2 );
Vec2d vec2d_sub( Vec2d v1, Vec2d v2 );
double vec2d_dot_product( Vec2d v1, Vec2d v2 );
Vec2d vec2d_times_scalar( Vec2d v, double a );
void vec2d_print( Vec2d v );

#endif /* _VEC_H_ */
