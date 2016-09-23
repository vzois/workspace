#ifndef HEIGHTMAP_H
#define HEIGHTMAP_H

#include <string>
#include "Types.h"
#include <noise/noise.h>
#include <vector>
#include <list>
#include "noiseutils.h"
#include <iostream>

class HeightMap
{
public:
	HeightMap(int w,int h);
	HeightMap():HeightMap(1024, 1024){};
	~HeightMap();

	int loadFromFile(std::string);/*Load HeightMap from File*/
	int generateWaves(GLint id, GLint f);/*generate height from noise*/
	int generateTerrain();/*generate height from noise*/
	int flat(int height);/*generate flat surface*/

	int heightValue(int x, int y);
	int initialize();

	int vertexBuffer();

	//Build geometry and texture buffers
	void texCoord(float x, float y, int size){ this->tex.push_back((float)x / (float)size);  this->tex.push_back(-(float)y / (float)size);}
	void vertCoord(int x, int y, int z){ this->vertices.push_back(x); this->vertices.push_back(y); this->vertices.push_back(z); }
	

	//Access vertices and texture coordinates
	std::vector<GLint>& getVertCoord(){ return this->vertices; }
	std::vector<GLfloat>& getTexCoord(){ return this->tex; }
	int getVerts(){ return this->verts; }

	GLfloat normalize(GLfloat x, GLfloat min, GLfloat max){ return (x - min) / (max - min); }

private:
	BYTE *heightmap;
	int height = 0;
	int width = 0;
	std::vector<GLint> vertices;
	std::vector<GLfloat> tex;
	int verts = 0;
	noise::module::Perlin pm;
	noise::module::Cylinders cm;
	noise::module::TranslatePoint tm;

	module::Add combM;
};

#endif