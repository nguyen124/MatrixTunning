#include "node.h"
#include <stdlib.h>

Node::Node(QuadTree* qt, Region rg) :qtree(qt), nodeRegion(rg)
{
	NE = NW = SE = SW = parent = NULL;
	isLeaf = true;
	//qtree->addNode(this);
}

Node::~Node()
{
	if (parent != NULL){
		delete parent;
	}
	if (NE != NULL){
		delete NE;
	}
	if (NW != NULL){
		delete NW;
	}
	if (SW != NULL){
		delete SW;
	}
	if (SE != NULL){
		delete SE;
	}
	if (qtree != NULL){
		delete qtree;
	}
	particlesInNode.clear();
}
bool Node::needSplit() {
	if (!isLeaf)
		return false;

	Node* neighbor;
	neighbor = findNorth();
	if (neighbor) {
		if (neighbor->SE)
		if (!neighbor->SE->isLeaf)
			return true;
		if (neighbor->SW)
		if (!neighbor->SW->isLeaf)
			return true;
	}

	neighbor = findSouth();
	if (neighbor) {
		if (neighbor->NE)
		if (!neighbor->NE->isLeaf)
			return true;
		if (neighbor->NW)
		if (!neighbor->NW->isLeaf)
			return true;
	}

	neighbor = findEast();
	if (neighbor) {
		if (neighbor->NW)
		if (!neighbor->NW->isLeaf)
			return true;
		if (neighbor->SW)
		if (!neighbor->SW->isLeaf)
			return true;
	}

	neighbor = findWest();
	if (neighbor) {
		if (neighbor->NE)
		if (!neighbor->NE->isLeaf)
			return true;
		if (neighbor->SE)
		if (!neighbor->SE->isLeaf)
			return true;
	}
	return false;
}

void Node::addNE() {
	float newX = nodeRegion.x + (float)nodeRegion.width / 2.;
	Node* ne = new Node(qtree,
		Region(newX, nodeRegion.y, nodeRegion.width / 2.,
		nodeRegion.height / 2.));
	//ne->isLeaf = true;	
	ne->parent = this;
	qtree->leaves.push_back(ne);
	this->NE = ne;
}

void Node::addNW() {
	Node* nw = new Node(qtree,
		Region(nodeRegion.x, nodeRegion.y, nodeRegion.width / 2.,
		nodeRegion.height / 2.));
	nw->parent = this;
	//nw->isLeaf = true;
	qtree->leaves.push_back(nw);
	this->NW = nw;
}

void Node::addSW() {
	float newY = nodeRegion.y + (float)nodeRegion.height / 2.;
	Node* sw = new Node(qtree,
		Region(nodeRegion.x, newY, nodeRegion.width / 2.,
		nodeRegion.height / 2.));
	sw->parent = this;
	//sw->isLeaf = true;
	qtree->leaves.push_back(sw);
	this->SW = sw;
}

void Node::addSE() {
	float newX = nodeRegion.x + (float)nodeRegion.width / 2.;
	float newY = nodeRegion.y + (float)nodeRegion.height / 2.;
	Node* se = new Node(qtree,
		Region(newX, newY, nodeRegion.width / 2.,
		nodeRegion.height / 2.));
	se->parent = this;
	//se->isLeaf = true;
	qtree->leaves.push_back(se);
	this->SE = se;
}

Node* Node::findNorth() {
	if (hasParent()) //it is not the head of the tree
	{
		if (this == parent->SW)
			return parent->NW;
		if (this == parent->SE)
			return parent->NE;
		Node* n = parent->findNorth();
		if (n != NULL)
		if (n->isLeaf)
			return n;
		else if (this == parent->NW)
			return n->SW;
		else
			return n->SE;
	}
	return NULL;
}

Node* Node::findSouth() {
	if (hasParent()) //it is not the head of the tree
	{
		if (this == parent->NW)
			return parent->SW;
		if (this == parent->NE)
			return parent->SE;
		Node* n = parent->findSouth();
		if (n)
		if (n->isLeaf)
			return n;
		else if (this == parent->SW)
			return n->NW;
		else
			return n->NE;
	}
	return NULL;
}

Node* Node::findEast() {
	if (hasParent()) //it is not the head of the tree
	{
		if (this == parent->NW)
			return parent->NE;
		if (this == parent->SW)
			return parent->SE;
		Node* n = parent->findEast();
		if (n)
		if (n->isLeaf)
			return n;
		else if (this == parent->NE)
			return n->NW;
		else
			return n->SW;
	}
	return NULL;
}

Node* Node::findWest() {
	if (hasParent()) //it is not the head of the tree
	{
		if (this == parent->NE)
			return parent->NW;
		if (this == parent->SE)
			return parent->SW;
		Node* n = parent->findWest();
		if (n)
		if (n->isLeaf)
			return n;
		else if (this == parent->NW)
			return n->NE;
		else
			return n->SE;
	}
	return NULL;
}
bool Node::hasParent() {
	return (parent != NULL);
}
;
bool Node::addParticle(particle_t* particle) {
	if (this->contains(particle)) {
		if (isLeaf && ((particlesInNode.size() < MAX_PARTICLES_PER_NODE)
			/*|| this->nodeRegion.width < (MAXRADIUS + 1)*/)) {
			particlesInNode.push_back(particle);
			//remove the actor from the previous cell
			if (particle->container) {
				//int idx = std::find(temp.begin(), temp.end(), particle);
				//if (idx > -1)
				vector<particle_t*> temp = particle->container->particlesInNode;
				temp.erase(
					std::remove(temp.begin(),
					temp.end(), particle), temp.end());
			}
			particle->container = this;
		}
		else {
			split();
			if (!NE->addParticle(particle))
			{
				if (!NW->addParticle(particle))
				{
					if (!SE->addParticle(particle))
					{
						SW->addParticle(particle);
					}
				}
			};
		}
		return true;
	}
	return false;
}
;
void Node::distributeParticles() {
	for (int i = 0; i < particlesInNode.size(); i++) {
		if (!NE->addParticle(particlesInNode[i])){
			if (!NW->addParticle(particlesInNode[i]))
			{
				if (!SE->addParticle(particlesInNode[i]))
				{
					SW->addParticle(particlesInNode[i]);
				}
			}
		};
	}
	particlesInNode.clear();
	//particlesInNode.shrink_to_fit();
}

Region Node::boundingRegion() const {
	return nodeRegion;
}

bool Node::contains(particle_t* particle) {

	if (
		((nodeRegion.x <= particle->x) && (particle->x < nodeRegion.x + nodeRegion.width)
		||
		(particle->x == 1 && nodeRegion.x + nodeRegion.width == 1))
		&& ((nodeRegion.y <= particle->y) && (particle->y < nodeRegion.y + nodeRegion.height)
		||
		(particle->y == 1 && nodeRegion.y + nodeRegion.height == 1))
		)
	{
		return true;
	}

	return false;
}
;
void Node::split() {
	if (isLeaf) {
		this->addNE();
		this->addNW();
		this->addSE();
		this->addSW();
		isLeaf = false;

		//	int idx = std::find(qtree->leaves.begin(), qtree->leaves.end(), this);
		//	if (idx > -1)
		//		qtree->leaves.erase(qtree->leaves.begin() + idx);
		qtree->leaves.erase(
			std::remove(qtree->leaves.begin(), qtree->leaves.end(), this),
			qtree->leaves.end());

	}
	distributeParticles();
}
