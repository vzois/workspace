#ifndef RENDER_H
#define RENDER_H

#include "Texture.h"
#include "HeightMap.h"
#include "Camera.h"
#include "WShader.h"

#define GL_CLAMP_TO_EDGE 0x812F

#define GL_TEXTURE0_ARB                     0x84C0
#define GL_TEXTURE1_ARB                     0x84C1
#define GL_TEXTURE2_ARB                     0x84C2
#define GL_TEXTURE3_ARB                     0x84C3
#define GL_TEXTURE4_ARB                     0x84C4
#define GL_TEXTURE5_ARB                     0x84C5

#define GL_COMBINE_ARB						0x8570
#define GL_RGB_SCALE_ARB					0x8573

typedef struct RendProp{
	RendProp(){};
	GLuint textures[MAX_TEXTURES];
	HeightMap *hmap[MAX_HEIGHTMAPS];
	Camera camera;
	WShader shader;
	WShader tts;
};

class Render
{
public:
	Render();
	~Render();
	void water_quad();
	void initializeGL();
	void loadScene();
	void renderScene();
	void resize(int w, int h);
	void animateScene();

	void renderEnvironment(bool enableCaustics);
	//SCENE OBJECTS
	void skybox(float x, float y, float z, float width, float height, float length);
	void terrain(HeightMap *hmap);

	void reflection();
	void refraction();
	void caustics();
	void water();
	void water_tesselation();
private:
	RendProp rp;
	unsigned int activeTerrainHmap;
};

#endif
