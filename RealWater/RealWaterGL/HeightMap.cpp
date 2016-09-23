#include "HeightMap.h"

HeightMap::HeightMap(int w, int h)
{
	this->width = w;
	this->height = h;
}


HeightMap::~HeightMap()
{

}

int HeightMap::initialize(){
	if (this->width <= 0) return ERROR;
	if (this->height <= 0) return ERROR;
	this->heightmap = new BYTE[this->width*this->height];
	if (this->heightmap == NULL) return ERROR;

	return SUCCESS;
}

int HeightMap::loadFromFile(std::string filename){
	FILE *fp = NULL;

	if (this->heightmap == NULL) return handleError("Error while opening ( " + filename + " ) = heightmap not allocated");

	fp = fopen(filename.c_str(), "rb");
	if (fp == NULL) return handleError("Error while opening ( " + filename + " )");

	//READ 1 BYTE FOR EACH PLACE IN THE HEIGHTMAP
	fread(this->heightmap,1,(this->width*this->height),fp);
	if (ferror(fp)) return handleError("Error while reading ( "+filename+" )");
	fclose(fp);

	return SUCCESS;
}

int HeightMap::generateTerrain(){//TODO
	/*GENERATE HEIGHTMAP WITH PERLIN NOISE*/
	if (this->heightmap == NULL) return handleError("Error generating Terrain, heightmap not allocated");
	//pm.SetFrequency(16);
	//pm.SetOctaveCount(1);
	//pm.SetPersistence(0.1);
	//pm.SetLacunarity(5);

	utils::NoiseMap hmap;
	utils::NoiseMapBuilderPlane hmap_builder;
	hmap_builder.SetSourceModule(pm);
	hmap_builder.SetDestNoiseMap(hmap);
	hmap_builder.SetDestSize(this->width, this->height);
	hmap_builder.SetBounds(9.0, 10.0, 1.0, 5.0);
	hmap_builder.Build();

	for (int i = 0; i < this->width; i++){
		for (int j = 0; j < this->height; j++){
			this->heightmap[j + i*this->width] = (BYTE)round((hmap.GetValue(i, j)*128 + 256 )/2);
		}
	}

	/*utils::RendererImage renderer;
	utils::Image image;
	renderer.SetSourceNoiseMap(hmap);
	renderer.SetDestImage(image);
	renderer.Render();

	utils::WriterBMP writer;
	writer.SetSourceImage(image);
	writer.SetDestFilename("tutorial.bmp");
	writer.WriteDestFile();*/
	
	return SUCCESS;
}

int HeightMap::generateWaves(GLint id, GLint f){//TODO
	if (this->heightmap == NULL) return handleError("Error generating Terrain, heightmap not allocated");
	pm.SetFrequency(0);
	pm.SetPersistence(0);
	pm.SetSeed(GetTickCount());
	cm.SetFrequency(0.1);

	combM.SetSourceModule(0, pm);
	combM.SetSourceModule(1, cm);

	tm.SetSourceModule(0, combM);
	
	int ss = 50;
	int ee = 93;
	for (int i = ss; i < ee; i++){
		tm.SetTranslation(i*0.5);
		utils::NoiseMap hmap;
		utils::NoiseMapBuilderPlane hmap_builder;
		hmap_builder.SetSourceModule(tm);
		hmap_builder.SetDestNoiseMap(hmap);
		hmap_builder.SetDestSize(this->width, this->height);
		hmap_builder.SetBounds(0.0, 16.0, 0.0, 16.0);
		hmap_builder.Build();

		GLfloat min = 0.0;
		GLfloat max = 0.0;
		GLfloat *values = new GLfloat[this->width*this->height];

		for (int i = 0; i < this->width; i++){
			for (int j = 0; j < this->height; j++){
				values[i*this->width + j] = hmap.GetValue(i, j);
				if (max < hmap.GetValue(i, j)) max = hmap.GetValue(i, j);
				if (min > hmap.GetValue(i, j)) min = hmap.GetValue(i, j);
			}
		}

		for (int i = 0; i < this->width; i++){
			for (int j = 0; j < this->height; j++){
				this->heightmap[j + i*this->width] = (BYTE)round(normalize(values[i*this->width + j], min, max) * WAVE_HEIGHT);
			}
		}
		delete[] values;

		utils::RendererImage renderer;
		utils::Image image;
		renderer.SetSourceNoiseMap(hmap);
		renderer.SetDestImage(image);
		renderer.Render();

		char szBuffer[255];
		sprintf(szBuffer, "%s%d%d.bmp", "heightmaps/waves/wave", (i - ss) / 10, (i - ss) % 10);
		utils::WriterBMP writer;
		writer.SetSourceImage(image);
		writer.SetDestFilename(std::string(szBuffer));
		writer.WriteDestFile();
	}
	return SUCCESS;
}

int HeightMap::flat(int height){
	/*FLAT TERRAIN*/
	if (this->heightmap == NULL) return handleError("Error generating flat surface, heightmap not allocated");
	for (int i = 0; i < this->height*this->width; i++){
		this->heightmap[i] = WATER_HEIGHT;
	}
	return SUCCESS;
}

int HeightMap::heightValue(int x, int y){
	/*GET HEIGHT VALUE FROM MAP*/
	int XX = abs(x%this->width);
	int YY = abs(y%this->height);
	return this->heightmap[YY + XX*this->width];
}

int HeightMap::vertexBuffer(){
	int x, y, z;
	if (this->heightmap == NULL) return handleError("Error generating vertex buffer, heightmap not allocated");
	bool _switch = false;
	this->vertices.reserve(2048);
	this->tex.reserve(2048);

	for (int i = 0; i <= TERRAIN_SIZE; i += TERRAIN_STEP){
		if (_switch){
			for (int j = TERRAIN_SIZE; j >= 0; j -= TERRAIN_STEP){
				x = i; y = this->heightValue(i, j); z = j;
				vertices.push_back(x); vertices.push_back(y); vertices.push_back(z);
				tex.push_back((float)x / (float)TERRAIN_SIZE); tex.push_back((float)z / (float)TERRAIN_SIZE);
				verts++;

				x = i + TERRAIN_STEP; y = this->heightValue(i + TERRAIN_STEP, j); z = j;
				vertices.push_back(x); vertices.push_back(y); vertices.push_back(z);
				tex.push_back((float)x / (float)TERRAIN_SIZE); tex.push_back((float)z / (float)TERRAIN_SIZE);
				verts++;
			}
		}
		else{
			for (int j = 0; j<= TERRAIN_SIZE; j += TERRAIN_STEP){
				x = i + TERRAIN_STEP; y = this->heightValue(i + TERRAIN_STEP, j); z = j;
				vertices.push_back(x); vertices.push_back(y); vertices.push_back(z);
				tex.push_back((float)x / (float)TERRAIN_SIZE); tex.push_back((float)z / (float)TERRAIN_SIZE);
				verts++;

				x = i; y = this->heightValue(i, j); z = j;
				vertices.push_back(x); vertices.push_back(y); vertices.push_back(z);
				tex.push_back((float)x / (float)TERRAIN_SIZE); tex.push_back((float)z / (float)TERRAIN_SIZE);
				verts++;
			}
		}
		_switch = !_switch;
	}
}
