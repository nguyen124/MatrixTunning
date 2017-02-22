#include <vector>
#include <algorithm>
#include "common.h"
#include "quadTree.h"
using namespace std;

#define MAX_PARTICLES_PER_NODE 2

struct Region {
	double x;
	double y;
	float width;
	float height;
	Region() {
		x = y = 0;
		width = height = 0;
	}
	;
	Region(double xCor, double yCor, float w, float h) {
		x = xCor;
		y = yCor;
		width = w;
		height = h;
	}
};

class Node {
public:
	Node* parent;
	Node* NE; //North East
	Node* NW; //Nort West
	Node* SW; //South West
	Node* SE; //South East
	bool isLeaf;

	QuadTree* qtree;
	Region* nodeRegion;
	vector<particle_t*> particlesInNode;

	void split();
	bool needSplit();
	bool hasParent();

	void addNE();
	void addNW();
	void addSW();
	void addSE();

	Node* findNorth();
	Node* findSouth();
	Node* findEast();
	Node* findWest();

	Node(QuadTree* qtree, Region* rg);

	bool addParticle(particle_t* p);
	void distributeParticles();
	void checkCollision(particle_t* p, int &comps);
	bool contains(particle_t* particle);
	Region* boundingRegion() const;
};
