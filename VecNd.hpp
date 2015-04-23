#ifndef _VECND_H_
#define _VECND_H_

#include <iostream>

template< int dim >
class VecNd
{
public:
    VecNd();
    VecNd( const VecNd& other );
    explicit VecNd( const bool initilize );
    explicit VecNd( const double &x );
    VecNd( const double &x, const double &y );
    VecNd( const double &x, const double &y, const double &z );

    double  operator[] ( const int index ) const;    
    double& operator[] ( const int index );

    // todo:: copy-constructor and assignment operators...
    // do them correctly
    VecNd<dim> &operator= ( const VecNd<dim> &other );
    VecNd<dim>  operator+ ( const VecNd<dim> &b ) const;
    VecNd<dim>  operator- ( const VecNd<dim> &b ) const;
    VecNd<dim>  operator- ( ) const;
    VecNd<dim>  operator* ( const double &c ) const;
    double operator* ( const VecNd<dim> &b ) const;    
    double square() const;

    double x() const;
    double y() const;
    double z() const;
    
    virtual ~VecNd() {};
private:
    double comp[dim];
};

template< int dim >
inline
VecNd<dim>::VecNd()
{}

template< int dim >
inline
VecNd<dim>::VecNd( const VecNd<dim> &other )
{
    for( int i = 0; i < dim; i++ )
	comp[i] = other.comp[i];
}

template< int dim >
inline
VecNd<dim>::VecNd( const bool initilize )
{
    if( dim == 1 ){
	comp[0] = 0.0;
    } else if( dim == 2 ){
	comp[0] = 0.0;
	comp[1] = 0.0;
    } else if( dim == 3 ){
	comp[0] = 0.0;
	comp[1] = 0.0;
	comp[2] = 0.0;
    } else {
	std::cout << "Unsupported dim in VecNd. Aborting" << std::endl;
	exit( EXIT_FAILURE );
    }
}

template< int dim >
inline
VecNd<dim>::VecNd( const double &x )
{
    if( dim == 1 ){
	comp[0] = x;
    } else {
	std::cout << "Number of arguments to VecNd constructor doesn't match dim. Aborting" << std::endl;
	exit( EXIT_FAILURE );
    }

}

template< int dim >
inline
VecNd<dim>::VecNd( const double &x, const double &y )
{
    if( dim == 2 ){
	comp[0] = x;
	comp[1] = y;
    } else {
	std::cout << "Number of arguments to VecNd constructor doesn't match dim. Aborting" << std::endl;
	exit( EXIT_FAILURE );
    }
}

template< int dim >
inline
VecNd<dim>::VecNd( const double &x, const double &y, const double &z )
{
    if( dim == 3 ){
	comp[0] = x;
	comp[1] = y;
	comp[2] = z;
    } else {
	std::cout << "Number of arguments to VecNd constructor doesn't match dim. Aborting" << std::endl;
	exit( EXIT_FAILURE );
    }
}

template< int dim >
inline
double VecNd<dim>::operator[] ( const int index ) const
{
    if( index >= 0 && index <= dim - 1 ){	
	return comp[ index ];
    } else {
	std::cout << "Index out of range in VecNd. Aborting" << std::endl;
	exit( EXIT_FAILURE );
    }
}

template< int dim >
inline
double &VecNd<dim>::operator[] ( const int index )
{
    if( index >= 0 && index <= dim - 1 ){	
	return comp[ index ];
    } else {
	std::cout << "Index out of range in VecNd. Aborting" << std::endl;
	exit( EXIT_FAILURE );
    }
}

template< int dim >
inline
VecNd<dim> &VecNd<dim>::operator= ( const VecNd<dim> &other )
{
    for( int i = 0; i < dim; i++ )
	comp[i] = other.comp[i];
    return *this;
}


template< int dim >
inline
VecNd<dim> VecNd<dim>::operator+ ( const VecNd<dim> &b ) const
{
    VecNd<dim> res;
    for( int i = 0; i < dim; i++ )
	res.comp[i] = comp[i] + b.comp[i];
    return res;
}

template< int dim >
inline
VecNd<dim> VecNd<dim>::operator- ( const VecNd<dim> &b ) const
{
    VecNd<dim> res;
    for( int i = 0; i < dim; i++ )
	res.comp[i] = comp[i] - b.comp[i];
    return res;
}

template< int dim >
inline
VecNd<dim> VecNd<dim>::operator- () const
{
    VecNd<dim> res;
    for( int i = 0; i < dim; i++ )
	res.comp[i] = -comp[i];
    return res;
}

template< int dim >
inline
VecNd<dim> VecNd<dim>::operator* ( const double &factor ) const
{
    VecNd<dim> res;
    for( int i = 0; i < dim; i++ )
	res.comp[i] = factor * comp[i];
    return res;
}

template< int dim >
inline
double VecNd<dim>::operator* ( const VecNd<dim> &b ) const
{
    double dot_product = 0.0;
    for( int i = 0; i < dim; i++ )
	dot_product += comp[i] * b.comp[i];
    return dot_product;
}

template< int dim >
inline
double VecNd<dim>::square ( ) const
{
    double self_dot_product = 0.0;
    for( int i = 0; i < dim; i++ )
	self_dot_product += comp[i] * comp[i];
    return self_dot_product;
}

template< int dim >
inline
double VecNd<dim>::x( ) const
{
    if( dim >= 1 && dim <= 3  ){
	return comp[0];
    } else {
	std::cout << "Unsupported dim in VecNd.x. Aborting" << std::endl;
	exit( EXIT_FAILURE );
    }
}

template< int dim >
inline
double VecNd<dim>::y( ) const
{
    if( dim >= 2 && dim <= 3  ){
	return comp[1];
    } else {
	std::cout << "Unsupported dim in VecNd.y. Aborting" << std::endl;
	exit( EXIT_FAILURE );
    }
}

template< int dim >
inline
double VecNd<dim>::z( ) const
{
    if( dim == 3 ){
	return comp[2];
    } else {
	std::cout << "Unsupported dim in VecNd.z. Aborting" << std::endl;
	exit( EXIT_FAILURE );
    }
}


// Global functions

template <int dim>
inline
VecNd<dim> operator* ( const double &factor, const VecNd<dim> &vec )
{
    return vec * factor;
}


template< int dim >
inline
std::ostream &operator<< ( std::ostream &out,  const VecNd<dim> &vec )
{
    out << "( ";
    for( int i = 0; i < dim - 1; i++ )
	out << vec[i] << ' ';
    out << vec[ dim - 1 ];
    out << " )";
    
    return out;
}

inline
std::ostream &operator<< ( std::ostream &out,  const VecNd<1> &vec )
{
    out << "( " << vec[0] << " )";
    
    return out;
}


#endif /* _VECND_H_ */
