#ifndef QUADTREE_H
#define QUADTREE_H

#include "common.h"
#include <vector>
using namespace std;

class QuadTree {
public:
	Node* root;
	vector<Node*> leaves;
	QuadTree(int n, particle_t* particles);
	~QuadTree();
	void balance();
	//detect collisions and return the amount of comparisons made
	int collisionDetect();
	void addNode(Node* n);

};

#endif
