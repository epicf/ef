#ifndef _NODE_REFERENCE_H_
#define _NODE_REFERENCE_H_

#include <iostream>
#include <vector>

class Node_reference{
public:
    int x, y, z;
public:
    Node_reference( int xx, int yy, int zz ): x(xx), y(yy), z(zz) {};
    virtual ~Node_reference() {};
    std::vector<Node_reference> adjacent_nodes() {
	Node_reference left( x+1, y, z );
	Node_reference right( x-1, y, z );
	Node_reference top( x, y+1, z );
	Node_reference bottom( x, y-1, z );
	Node_reference near( x, y, z-1 );
	Node_reference far( x, y, z+1 );
	std::vector<Node_reference> neighbours( {left, right, top, bottom, near, far} );
	return neighbours;
    };
    void print() {
	std::cout << "Node = " << x << " " << y << " " << z << std::endl;
    };
    bool at_domain_edge( int nx, int ny, int nz ){
	return ( x <= 0 || x >= nx - 1 ||
		 y <= 0 || y >= ny - 1 ||
		 z <= 0 || z >= nz - 1 );
    };
    bool left_from( Node_reference other_node ){
	return x-1 == other_node.x;
    };
    bool right_from( Node_reference other_node ){
	return x+1 == other_node.x;
    };
    bool top_from( Node_reference other_node ){	
	return y-1 == other_node.y;
    };
    bool bottom_from( Node_reference other_node ){
	return y+1 == other_node.y;
    };
    bool far_from( Node_reference other_node ){
	return z-1 == other_node.z;
    };
    bool near_from( Node_reference other_node ){
	return z+1 == other_node.z;
    };

    // comparison operators are necessary for std::sort and std::unique to work
    bool operator< ( const Node_reference &other_node ) const {
	return ( x < other_node.x ) ||
	       ( x == other_node.x && y < other_node.y ) ||
	       ( x == other_node.x && y == other_node.y && z < other_node.z );
    };
    bool operator== ( const Node_reference &other_node ) const {
	return x == other_node.x && y == other_node.y && z == other_node.z;
    };
    
};

#endif /* _NODE_REFERENCE_H_ */
