#include <stdlib.h>
#include "quadTree.h"
#include "common.h"
#include "node.h"
using namespace std;



QuadTree::QuadTree(int n, particle_t* particles) {
	//
	//int firstNodeSize = (size / cutoff) + 1;
	//root node is the whole first grid before dividing
	
	root = new Node(this, Region(0, 0, 1, 1));
	leaves.push_back(root);
	for (int i = 0; i < n; i++) {
		root->addParticle(&particles[i]);
	}
	//root is the first leaf
	
}
;

QuadTree::~QuadTree(){
	if (root != NULL){
		delete root;
	}
	leaves.clear();
}
//leaves.push_back(this);

void QuadTree::addNode(Node* node) {
	leaves.push_back(node);
}
void QuadTree::balance() {
	vector<Node*> tempLeaves = leaves;
	while (tempLeaves.size() > 0) {
		Node* leaf = tempLeaves.back();
		tempLeaves.pop_back();

		if (leaf->needSplit()) {
			leaf->split();
			//insert new leaves on the list
			tempLeaves.push_back(leaf->NE);
			tempLeaves.push_back(leaf->NW);
			tempLeaves.push_back(leaf->SE);
			tempLeaves.push_back(leaf->SW);
			//checks if its neighbors need to be split
			Node* test = leaf->findNorth();
			if (test)
				if ((test != leaf) && test->needSplit()) {
					tempLeaves.push_back(test);
				}

			test = leaf->findEast();
			if (test)
				if (test->needSplit()) {
					tempLeaves.push_back(test);
				}

			test = leaf->findWest();
			if (test)
				if (test->needSplit()) {
					tempLeaves.push_back(test);
				}

			test = leaf->findSouth();
			if (test)
				if (test->needSplit()) {
					tempLeaves.push_back(test);
				}
		}
	}
}
//detect collisions and return the amount of comparisons made
//int collisionDetect();
//void balance();
