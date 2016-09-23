#include "Render.h"

/*Water Properties*/
Vector3 waterColor(0.1f, 0.2f, 0.4f);
Vector3 lightPos_0(100.0f, 150.0f, 100.0f);
Vector3 lightPos_1(100.0f, -150.0f, 200.0f);
Vector3 lightPos_2(250.0f, 50.0f, -100.0f);
Vector3 lightPos_3(250.0f, 250.0f, 50.0f);
Vector3 lightPos_4(50.0f, 50.0f, -100.0f);

/*Water Input Data Flags*/
#define REFLECTION "reflectTex"
#define REFRACTION "refractTex"
#define NORMAL_MAP "normalMap"
#define DUDV_MAP "dudvMap"
#define DEPTH_MAP "depthMap"
#define DISP_MAP "dispMap"

#define WATER_COLOR "waterColor"
#define CAMERA_POS "cameraPos"
#define LIGHT_POS_0 "lightPos_0"
#define LIGHT_POS_1 "lightPos_1"
#define LIGHT_POS_2 "lightPos_2"
#define LIGHT_POS_3 "lightPos_3"
#define LIGHT_POS_4 "lightPos_4"

#define VIEW_PROJ_MATRIX "gVP"

#define WATER_BOUNDARY_MIN 0.0f
#define WATER_BOUNDARY_MAX 1000.0f

GLuint waterID;
std::vector<GLfloat> waterGeom;
std::vector<GLfloat> waterGeom2;
std::vector<GLfloat> waterGeom3;
std::vector<GLfloat> waterTex;

Render::Render(){}

Render::~Render(){}

void Render::initializeGL(){//TODO
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_TEXTURE_2D);
	glEnable(GL_CULL_FACE);

	this->activeTerrainHmap = TERRAIN_HMAP;
	this->rp.camera.position(Vector3(375, 102, 321),
		Vector3(354, 72, 300), Vector3(0, 1, 0));

	GLint GlewInitResult = glewInit();
	if (GLEW_OK != GlewInitResult)
	{
		printf("ERROR: %s\n", glewGetErrorString(GlewInitResult));
		exit(EXIT_FAILURE);
	}
	rp.shader.initShader(false);
	rp.shader.off();

	rp.tts.initShader(true);
	rp.tts.off();
}

void Render::water_quad(){
	FILE *infile;
	std::string file = "quad.asc";
	char dummy[256];
	if ((infile = fopen(file.c_str(), "r")) == NULL){
		handleError("Error loading water quad file ( "+ file+" )");
	}
	//waterGeom2.reserve(3000);
	float x, y, z;
	GLfloat u, v;
	while (fscanf(infile, "%s", dummy) == 1) {
		fscanf(infile, "%f %f %f %f %f", &x, &y, &z, &u, &v);
		waterGeom2.push_back(x); waterGeom2.push_back(WATER_HEIGHT); waterGeom2.push_back(z);
		fscanf(infile, "%f %f %f %f %f", &x, &y, &z, &u, &v);
		waterGeom2.push_back(x); waterGeom2.push_back(WATER_HEIGHT); waterGeom2.push_back(z);
		fscanf(infile, "%f %f %f %f %f", &x, &y, &z, &u, &v);
		waterGeom2.push_back(x); waterGeom2.push_back(WATER_HEIGHT); waterGeom2.push_back(z);
	}

	if (fclose(infile)) handleError("Error closing water quad file ( " + file + " )");
	
}

void Render::loadScene(){
	//WATER GEOMETRY
	waterGeom.push_back(WATER_BOUNDARY_MIN); waterGeom.push_back(WATER_HEIGHT); waterGeom.push_back(WATER_BOUNDARY_MIN);
	waterGeom.push_back(WATER_BOUNDARY_MIN); waterGeom.push_back(WATER_HEIGHT); waterGeom.push_back(WATER_BOUNDARY_MAX);
	waterGeom.push_back(WATER_BOUNDARY_MAX); waterGeom.push_back(WATER_HEIGHT); waterGeom.push_back(WATER_BOUNDARY_MIN);
	waterGeom.push_back(WATER_BOUNDARY_MAX); waterGeom.push_back(WATER_HEIGHT); waterGeom.push_back(WATER_BOUNDARY_MAX);
	waterGeom.push_back(WATER_BOUNDARY_MAX); waterGeom.push_back(WATER_HEIGHT); waterGeom.push_back(WATER_BOUNDARY_MIN);
	waterGeom.push_back(WATER_BOUNDARY_MIN); waterGeom.push_back(WATER_HEIGHT); waterGeom.push_back(WATER_BOUNDARY_MAX);

	water_quad();

	//LOAD SKYBOX PROPERTIES
	Texture *tex= NULL;
	std::string path;

	tex = new Texture;
	path = BOX_TEX_PATH"Top.bmp";
	tex->loadTextureFromFile(path);
	rp.textures[BOX_TOP] = tex->getTextureBinding();
	delete tex;

	tex = new Texture;
	path = BOX_TEX_PATH"Bottom.bmp";
	tex->loadTextureFromFile(path);
	rp.textures[BOX_BOTTOM] = tex->getTextureBinding();
	delete tex;

	tex = new Texture;
	path = BOX_TEX_PATH"Front.bmp";
	tex->loadTextureFromFile(path);
	rp.textures[BOX_FRONT] = tex->getTextureBinding();
	delete tex;

	tex = new Texture;
	path = BOX_TEX_PATH"Back.bmp";
	tex->loadTextureFromFile(path);
	rp.textures[BOX_BACK] = tex->getTextureBinding();
	delete tex;

	tex = new Texture;
	path = BOX_TEX_PATH"Left.bmp";
	tex->loadTextureFromFile(path);
	rp.textures[BOX_LEFT] = tex->getTextureBinding();
	delete tex;

	tex = new Texture;
	path = BOX_TEX_PATH"Right.bmp";
	tex->loadTextureFromFile(path);
	rp.textures[BOX_RIGHT] = tex->getTextureBinding();
	delete tex;

	//LOAD TERRAIN TEXTURES
	tex = new Texture;
	tex->loadTextureFromFile(TERRAIN_TEX_PATH);
	rp.textures[TERRAIN_TEX] = tex->getTextureBinding();
	delete tex;

	tex = new Texture;
	tex->loadTextureFromFile(DETAIL_TERRAIN_TEX_PATH);
	rp.textures[DETAIL_TERRAIN_TEX] = tex->getTextureBinding();
	delete tex;

	//TERRAIN HEIGHTMAP
	rp.hmap[TERRAIN_HMAP] = new HeightMap(TERRAIN_SIZE, TERRAIN_SIZE);
	rp.hmap[TERRAIN_HMAP]->initialize();
	rp.hmap[TERRAIN_HMAP]->loadFromFile(TERRAIN_HMAP_PATH);
	rp.hmap[TERRAIN_HMAP]->vertexBuffer();

	rp.hmap[PROC_TERRAIN_01] = new HeightMap(TERRAIN_SIZE, TERRAIN_SIZE);
	rp.hmap[PROC_TERRAIN_01]->initialize();
	rp.hmap[PROC_TERRAIN_01]->generateTerrain();
	rp.hmap[PROC_TERRAIN_01]->vertexBuffer();

	//WATER TEXTURE
	tex = new Texture;
	tex->loadTextureFromFile(WATER_TEX_PATH);
	rp.textures[WATER_TEX] = tex->getTextureBinding();
	delete tex;

	tex = new Texture;
	tex->loadTextureFromFile(NORMAL_MAP_TEX_PATH);
	rp.textures[NORMAL_MAP_TEX] = tex->getTextureBinding();
	delete tex;

	tex = new Texture;
	tex->loadTextureFromFile(DUDV_MAP_TEX_PATH);
	rp.textures[DUDV_MAP_TEX] = tex->getTextureBinding();
	delete tex;

	tex = new Texture;
	tex->renderTexture(TEXTURE_SIZE, 3, GL_RGB);
	rp.textures[REFLECTION_TEX] = tex->getTextureBinding();
	delete tex;

	tex = new Texture;
	tex->renderTexture(TEXTURE_SIZE, 3, GL_RGB);
	rp.textures[REFRACTION_TEX] = tex->getTextureBinding();
	delete tex;

	tex = new Texture;
	tex->renderTexture(TEXTURE_SIZE, 1, GL_DEPTH_COMPONENT);
	rp.textures[DEPTH_TEX] = tex->getTextureBinding();
	delete tex;

	//CAUSTICS
	char szBuffer[255];

	for (int i = 0; i < CAUSTIC_NUM; i++){
		tex = new Texture;
		sprintf(szBuffer, "%s%d%d.bmp", "caust", i / 10, i % 10);
		tex->loadTextureFromFile(CAUSTIC_PATH + std::string(szBuffer));
		rp.textures[CAUSTIC_OFFSET + i] = tex->getTextureBinding();
		delete tex;
	}

	//WATER HEIGHTMAPS
	for (int i = 0; i < WAVE_HMAP_NUM; i++){
		tex = new Texture;
		sprintf(szBuffer, "%s%d%d.bmp", "wave", i / 10, i % 10);
		tex->loadTextureFromFile(WAVE_TEX_PATH + std::string(szBuffer));
		rp.textures[WAVE_OFFSET + i] = tex->getTextureBinding();
	}

}

void Render::renderScene(){
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glLoadIdentity();
	rp.camera.look();
	skybox(500, 200, 500, 2000, 2000, 2000);
	glutSwapBuffers();
}

void Render::resize(int w, int h){
	h = h == 0 ? 1 : h;
	float ratio = w / h;

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glViewport(0, 0, w, h);
	gluPerspective(DEFAULT_FOV, ratio, DEFAULT_NEAR, DEFAULT_FAR);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
}

void Render::animateScene(){
	if (frameRate(60)){
		//RENDER WATER
		this->reflection();
		this->refraction();

		glViewport(0, 0, WINDOW_SIZEX, WINDOW_SIZEY);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glLoadIdentity();

		Vector3 pos = rp.camera.getPos();
		rp.camera.collisionAvoidance(rp.hmap[this->activeTerrainHmap]->heightValue(pos[X], pos[Z]));
		bool tessellation = rp.camera.update();
		rp.camera.look();

		//RENDER TERRAIN & SKYBOX
		this->renderEnvironment(true);

		//RENDER WATER SURFACE
		glDisable(GL_CULL_FACE);
		
		if (tessellation) water_tesselation();
		else water();
		glEnable(GL_CULL_FACE);
		
		glutSwapBuffers();
	}
	else{
		Sleep(FRAME_DELAY);
	}
}

void Render::renderEnvironment(bool enableCaustics){
	if (enableCaustics) this->caustics();

	double terrainPlane[4] = { 0.0, 1.0, 0.0, -WATER_HEIGHT };
	glEnable(GL_CLIP_PLANE0);
	glClipPlane(GL_CLIP_PLANE0, terrainPlane);
	terrain(rp.hmap[this->activeTerrainHmap]);
	glDisable(GL_CLIP_PLANE0);
	skybox(500, 0, 500, 2000, 2000, 2000);
}

void Render::skybox(float x, float y, float z, float width, float height, float length){
	x = x - width / 2;
	y = y - height / 2;
	z = z - length / 2;

	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, rp.textures[BOX_BACK]);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glBegin(GL_QUADS);
	//BACK
	//glColor3f(0.0, 0.0, 1.0);
	glTexCoord2f(1.0f, 0.0f); glVertex3f(x + width, y, z);
	glTexCoord2f(1.0f, 1.0f); glVertex3f(x + width, y + height, z);
	glTexCoord2f(0.0f, 1.0f); glVertex3f(x, y + height, z);
	glTexCoord2f(0.0f, 0.0f); glVertex3f(x, y, z);
	glEnd();

	glBindTexture(GL_TEXTURE_2D, rp.textures[BOX_FRONT]);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glBegin(GL_QUADS);
	//FRONT
	//glColor3f(1.0, 0.0, 1.0);
	glTexCoord2f(1.0f, 0.0f); glVertex3f(x, y, z + length);
	glTexCoord2f(1.0f, 1.0f); glVertex3f(x, y + height, z + length);
	glTexCoord2f(0.0f, 1.0f); glVertex3f(x + width, y + height, z + length);
	glTexCoord2f(0.0f, 0.0f); glVertex3f(x + width, y, z + length);
	glEnd();

	glBindTexture(GL_TEXTURE_2D, rp.textures[BOX_TOP]);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glBegin(GL_QUADS);
	//TOP
	//glColor3f(0.0, 1.0, 1.0);
	glTexCoord2f(0.0f, 1.0f); glVertex3f(x + width, y + height, z);
	glTexCoord2f(0.0f, 0.0f); glVertex3f(x + width, y + height, z + length);
	glTexCoord2f(1.0f, 0.0f); glVertex3f(x, y + height, z + length);
	glTexCoord2f(1.0f, 1.0f); glVertex3f(x, y + height, z);
	glEnd();

	glBindTexture(GL_TEXTURE_2D, rp.textures[BOX_BOTTOM]);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glBegin(GL_QUADS);
	//BOTTOM
	//glColor3f(0.0, 1.0, 0.0);
	glTexCoord2f(1.0f, 0.0f); glVertex3f(x, y, z);
	glTexCoord2f(1.0f, 1.0f); glVertex3f(x, y, z + length);
	glTexCoord2f(0.0f, 1.0f); glVertex3f(x + width, y, z + length);
	glTexCoord2f(0.0f, 0.0f); glVertex3f(x + width, y, z);
	glEnd();

	glBindTexture(GL_TEXTURE_2D, rp.textures[BOX_LEFT]);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glBegin(GL_QUADS);
	//LEFT
	//glColor3f(1.0, 0.0, 0.0);
	glTexCoord2f(1.0f, 1.0f); glVertex3f(x, y + height, z);
	glTexCoord2f(0.0f, 1.0f); glVertex3f(x, y + height, z + length);
	glTexCoord2f(0.0f, 0.0f); glVertex3f(x, y, z + length);
	glTexCoord2f(1.0f, 0.0f); glVertex3f(x, y, z);
	glEnd();

	glBindTexture(GL_TEXTURE_2D, rp.textures[BOX_RIGHT]);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glBegin(GL_QUADS);
	//RIGHT
	//glColor3f(1.0, 1.0, 0.0);
	glTexCoord2f(0.0f, 0.0f); glVertex3f(x + width, y, z);
	glTexCoord2f(1.0f, 0.0f); glVertex3f(x + width, y, z + length);
	glTexCoord2f(1.0f, 1.0f); glVertex3f(x + width, y + height, z + length);
	glTexCoord2f(0.0f, 1.0f); glVertex3f(x + width, y + height, z);
	glEnd();
}

void Render::terrain(HeightMap *hmap){
	//INITIAL TEXTURE
	glActiveTexture(GL_TEXTURE0_ARB);
	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, rp.textures[TERRAIN_TEX]);

	//Enable Detailed Texture
	glActiveTexture(GL_TEXTURE1_ARB);
	glEnable(GL_TEXTURE_2D);
	glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_COMBINE_ARB);
	glTexEnvi(GL_TEXTURE_ENV, GL_RGB_SCALE_ARB, 2);
	glBindTexture(GL_TEXTURE_2D, rp.textures[DETAIL_TERRAIN_TEX]);
	glMatrixMode(GL_TEXTURE);
	glLoadIdentity();
	glScalef((float)DETAIL_SCALE, (float)DETAIL_SCALE, 1);
	glMatrixMode(GL_MODELVIEW);

	//Enable Texture Array for the corresponding Texture
	glClientActiveTexture(GL_TEXTURE0_ARB);
	glEnableClientState(GL_TEXTURE_COORD_ARRAY);
	glTexCoordPointer(2, GL_FLOAT, 0, &hmap->getTexCoord()[0]);

	//Enable Texture Array for the corresponding Texture
	glClientActiveTexture(GL_TEXTURE1_ARB);
	glEnableClientState(GL_TEXTURE_COORD_ARRAY);
	glTexCoordPointer(2, GL_FLOAT, 0, &hmap->getTexCoord()[0]);

	//Enable Vertex Array
	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(3, GL_INT, 0, &hmap->getVertCoord()[0]);
	
	//Draw Arrays as a Triangle Strip
	glDrawArrays(GL_TRIANGLE_STRIP, 0, hmap->getVerts());
	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_TEXTURE_COORD_ARRAY);
	
	//Disable Textures
	glActiveTexture(GL_TEXTURE1_ARB);
	glDisable(GL_TEXTURE_2D);
	glActiveTexture(GL_TEXTURE0_ARB);
	glDisable(GL_TEXTURE_2D);
}

void Render::reflection(){
	glViewport(0, 0, TEXTURE_SIZE, TEXTURE_SIZE);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glLoadIdentity();
	rp.camera.look();

	glPushMatrix();
	if (rp.camera.getPos()[Y] > WATER_HEIGHT){
		glTranslatef(0.0f, WATER_HEIGHT*2.0, 0.0f);
		glScalef(1.0, -1.0, 1.0);
		glCullFace(GL_FRONT);
		GLdouble clip[4] = { 0.0, 1.0, 0.0, -WATER_HEIGHT };
		glEnable(GL_CLIP_PLANE0);
		glClipPlane(GL_CLIP_PLANE0, clip);

		this->renderEnvironment(false);
		glDisable(GL_CLIP_PLANE0);
		glCullFace(GL_BACK);
	}
	else{
		GLdouble clip[4] = { 0.0, 1.0, 0.0, WATER_HEIGHT };
		glEnable(GL_CLIP_PLANE0);
		glClipPlane(GL_CLIP_PLANE0, clip);
		this->renderEnvironment(true);
		glDisable(GL_CLIP_PLANE0);
	}
	glPopMatrix();
	glBindTexture(GL_TEXTURE_2D, rp.textures[REFLECTION_TEX]);
	glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 0, 0, TEXTURE_SIZE, TEXTURE_SIZE);
}

void Render::refraction(){
	glViewport(0, 0, TEXTURE_SIZE, TEXTURE_SIZE);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glLoadIdentity();
	rp.camera.look();

	glPushMatrix();
	if (rp.camera.getPos()[Y] > WATER_HEIGHT){
		GLdouble clip[4] = { 0.0, -1.0, 0.0, WATER_HEIGHT };
		glEnable(GL_CLIP_PLANE0);
		glClipPlane(GL_CLIP_PLANE0, clip);
		this->renderEnvironment(true);
		glDisable(GL_CLIP_PLANE0);
	}
	else{
		glCullFace(GL_FRONT);
		GLdouble clip[4] = { 0.0, 1.0, 0.0, -WATER_HEIGHT };
		glEnable(GL_CLIP_PLANE0);
		glClipPlane(GL_CLIP_PLANE0, clip);
		this->renderEnvironment(true);
		glDisable(GL_CLIP_PLANE0);
		glCullFace(GL_BACK);
	}
	glPopMatrix();

	glBindTexture(GL_TEXTURE_2D, rp.textures[REFRACTION_TEX]);
	glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 0, 0, TEXTURE_SIZE, TEXTURE_SIZE);

	glBindTexture(GL_TEXTURE_2D, rp.textures[DEPTH_TEX]);
	glCopyTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, 0, 0, TEXTURE_SIZE, TEXTURE_SIZE, 0);
}

void Render::caustics(){
	glActiveTexture(GL_TEXTURE2_ARB);
	glEnable(GL_TEXTURE_2D);

	static int index = 0;
	glBindTexture(GL_TEXTURE_2D, rp.textures[index + CAUSTIC_OFFSET]);

	glClientActiveTexture(GL_TEXTURE2_ARB);
	glEnableClientState(GL_TEXTURE_COORD_ARRAY);
	glTexCoordPointer(2, GL_FLOAT, 0, &rp.hmap[this->activeTerrainHmap]->getTexCoord()[0]);

	static int frames = 0;
	if (frames % 5 == 0){ index = ((index + 1) % CAUSTIC_NUM); frames == 0; }
	frames++;

	glMatrixMode(GL_TEXTURE);
	glLoadIdentity();
	glScalef(CAUSTIC_SCALE, CAUSTIC_SCALE, 1);
	glMatrixMode(GL_MODELVIEW);

	double plane[4] = { 0.0, -1.0, 0.0, WATER_HEIGHT };
	glEnable(GL_CLIP_PLANE0);
	glClipPlane(GL_CLIP_PLANE0, plane);
	terrain(rp.hmap[TERRAIN_HMAP]);
	glDisable(GL_CLIP_PLANE0);
	glMatrixMode(GL_TEXTURE);
	glLoadIdentity();
	glMatrixMode(GL_MODELVIEW);
	glActiveTextureARB(GL_TEXTURE2_ARB);
	glDisable(GL_TEXTURE_2D);
}

void Render::water(){
	rp.shader.on();
	glActiveTexture(GL_TEXTURE0_ARB);
	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, rp.textures[REFLECTION_TEX]);

	glActiveTexture(GL_TEXTURE1_ARB);
	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, rp.textures[REFRACTION_TEX]);

	glActiveTexture(GL_TEXTURE2_ARB);
	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, rp.textures[NORMAL_MAP_TEX]);

	glActiveTexture(GL_TEXTURE3_ARB);
	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, rp.textures[DUDV_MAP_TEX]);

	glActiveTexture(GL_TEXTURE4_ARB);
	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, rp.textures[DEPTH_TEX]);

	rp.shader.setMatrix(rp.shader.getLocation(VIEW_PROJ_MATRIX), getVP(rp.camera.getPos(), rp.camera.getView(), rp.camera.getUp()));
	rp.shader.setVal(rp.shader.getLocation(REFLECTION), 0);
	rp.shader.setVal(rp.shader.getLocation(REFRACTION), 1);
	rp.shader.setVal(rp.shader.getLocation(NORMAL_MAP), 2);
	rp.shader.setVal(rp.shader.getLocation(DUDV_MAP), 3);
	rp.shader.setVal(rp.shader.getLocation(DEPTH_MAP), 4);

	rp.shader.setVector(rp.shader.getLocation(WATER_COLOR), waterColor, 1.0);
	rp.shader.setVector(rp.shader.getLocation(LIGHT_POS_0), lightPos_0, 1.0);
	rp.shader.setVector(rp.shader.getLocation(CAMERA_POS), rp.camera.getPos(), 1.0);

	static float move = 0.0f;
	float move2 = move * NORMAL_MAP_SCALE;
	float refrUV = WATER_UV;
	float normalUV = WATER_UV * NORMAL_MAP_SCALE;
	move += WATER_FLOW;

	
	std::vector<GLfloat> tex0, tex1, tex2, tex3, tex4;
	//TEX0
	tex0.push_back(0.0f); tex0.push_back(WATER_UV); tex0.push_back(0.0f); tex0.push_back(0.0f); tex0.push_back(WATER_HEIGHT); tex0.push_back(WATER_UV);
	tex0.push_back(WATER_UV); tex0.push_back(0.0f); tex0.push_back(WATER_HEIGHT); tex0.push_back(WATER_UV); tex0.push_back(0.0f); tex0.push_back(0.0f);
	glClientActiveTexture(GL_TEXTURE0_ARB);
	glEnableClientState(GL_TEXTURE_COORD_ARRAY);
	glTexCoordPointer(2, GL_FLOAT, 0, &tex0[0]);

	//TEX1
	tex1.push_back(0.0f); tex1.push_back(refrUV - move); tex1.push_back(0.0f); tex1.push_back(0.0f - move); tex1.push_back(refrUV); tex1.push_back(refrUV - move);
	tex1.push_back(refrUV); tex1.push_back(0.0f - move); tex1.push_back(refrUV); tex1.push_back(refrUV - move); tex1.push_back(0.0f); tex1.push_back(0.0f - move);
	glClientActiveTexture(GL_TEXTURE1_ARB);
	glEnableClientState(GL_TEXTURE_COORD_ARRAY);
	glTexCoordPointer(2, GL_FLOAT, 0, &tex1[0]);

	//TEX2
	tex2.push_back(0.0f); tex2.push_back(normalUV + move2); tex2.push_back(0.0f); tex2.push_back(0.0f + move2); tex2.push_back(normalUV); tex2.push_back(0.0f + move2);
	tex2.push_back(normalUV); tex2.push_back(normalUV + move2); tex2.push_back(normalUV); tex2.push_back(0.0f + move2); tex2.push_back(0.0f); tex2.push_back(0.0f + move2);
	glClientActiveTexture(GL_TEXTURE2_ARB);
	glEnableClientState(GL_TEXTURE_COORD_ARRAY);
	glTexCoordPointer(2, GL_FLOAT, 0, &tex2[0]);

	//TEX3
	tex3.push_back(0.0f); tex3.push_back(0.0f); tex3.push_back(0.0f); tex3.push_back(0.0f); tex3.push_back(0.0f); tex3.push_back(0.0f);
	tex3.push_back(0.0f); tex3.push_back(0.0f); tex3.push_back(0.0f); tex3.push_back(0.0f); tex3.push_back(0.0f); tex3.push_back(0.0f);
	glClientActiveTexture(GL_TEXTURE3_ARB);
	glEnableClientState(GL_TEXTURE_COORD_ARRAY);
	glTexCoordPointer(2, GL_FLOAT, 0, &tex3[0]);

	//TEX4
	tex4.push_back(0.0f); tex4.push_back(0.0f); tex4.push_back(0.0f); tex4.push_back(0.0f); tex4.push_back(0.0f); tex4.push_back(0.0f);
	tex4.push_back(0.0f); tex4.push_back(0.0f); tex4.push_back(0.0f); tex4.push_back(0.0f); tex4.push_back(0.0f); tex4.push_back(0.0f);
	glClientActiveTexture(GL_TEXTURE4_ARB);
	glEnableClientState(GL_TEXTURE_COORD_ARRAY);
	glTexCoordPointer(2, GL_FLOAT, 0, &tex4[0]);

	//TWO TRIANGLE FOR WATER SURFACE//
	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(3, GL_FLOAT, 0, &waterGeom[0]);
	glDrawArrays(GL_TRIANGLES, 0, 6);
	glDisableClientState(GL_VERTEX_ARRAY);

	glDisableClientState(GL_TEXTURE_COORD_ARRAY);
	glActiveTexture(GL_TEXTURE0_ARB);
	glDisable(GL_TEXTURE_2D);
	glActiveTexture(GL_TEXTURE1_ARB);
	glDisable(GL_TEXTURE_2D);
	glActiveTexture(GL_TEXTURE2_ARB);
	glDisable(GL_TEXTURE_2D);
	glActiveTexture(GL_TEXTURE3_ARB);
	glDisable(GL_TEXTURE_2D);
	glActiveTexture(GL_TEXTURE4_ARB);
	glDisable(GL_TEXTURE_2D);
	rp.shader.off();
}

void Render::water_tesselation(){
	static bool tess = true;
	int tessLevels = 0;
	if (tess){
		tess = false;
		glGetIntegerv(GL_MAX_TESS_GEN_LEVEL, &tessLevels);
		std::cout << "Max Tessellation Levels: " << tessLevels << std::endl;
	}

	static int frame = 0;
	static int wave = 0;
	if (frame == 2){ wave = (wave + 1) % WAVE_HMAP_NUM; frame = 0; }
	frame++;

	rp.tts.on();
	glActiveTexture(GL_TEXTURE0_ARB);
	glBindTexture(GL_TEXTURE_2D, rp.textures[REFLECTION_TEX]);

	glActiveTexture(GL_TEXTURE1_ARB);
	glBindTexture(GL_TEXTURE_2D, rp.textures[REFRACTION_TEX]);

	glActiveTexture(GL_TEXTURE2_ARB);
	glBindTexture(GL_TEXTURE_2D, rp.textures[NORMAL_MAP_TEX]);

	glActiveTexture(GL_TEXTURE3_ARB);
	glBindTexture(GL_TEXTURE_2D, rp.textures[DUDV_MAP_TEX]);

	glActiveTexture(GL_TEXTURE4_ARB);
	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, rp.textures[DEPTH_TEX]);

	glActiveTexture(GL_TEXTURE5_ARB);
	glBindTexture(GL_TEXTURE_2D, rp.textures[WAVE_OFFSET + wave]);

	rp.tts.setVal(rp.tts.getLocation(REFLECTION), 0);
	rp.tts.setVal(rp.tts.getLocation(REFRACTION), 1);
	rp.tts.setVal(rp.tts.getLocation(NORMAL_MAP), 2);
	rp.tts.setVal(rp.tts.getLocation(DUDV_MAP), 3);
	rp.tts.setVal(rp.tts.getLocation(DEPTH_MAP), 4);
	rp.tts.setVal(rp.tts.getLocation(DISP_MAP), 5);

	rp.tts.setVector(rp.tts.getLocation(WATER_COLOR), waterColor, 1.0);
	rp.tts.setVector(rp.tts.getLocation(LIGHT_POS_0), lightPos_0, 1.0);
	rp.tts.setVector(rp.tts.getLocation(CAMERA_POS), rp.camera.getPos(), 1.0);

	std::vector<GLfloat> tex0, tex1, tex2, tex3, tex4;
	static float move = 0.0f;
	float move2 = move * NORMAL_MAP_SCALE;
	float refrUV = WATER_UV;
	float normalUV = WATER_UV * NORMAL_MAP_SCALE;
	move += WATER_FLOW;

	int tris = 1324;
	//TEX0
	for (int i = 0; i < tris; i++){
		tex0.push_back(0.0f); tex0.push_back(WATER_UV); tex0.push_back(0.0f); tex0.push_back(0.0f); tex0.push_back(WATER_HEIGHT); tex0.push_back(WATER_UV);
		tex0.push_back(WATER_UV); tex0.push_back(0.0f); tex0.push_back(WATER_HEIGHT); tex0.push_back(WATER_UV); tex0.push_back(0.0f); tex0.push_back(0.0f);
	}
	glClientActiveTexture(GL_TEXTURE0_ARB);
	glEnableClientState(GL_TEXTURE_COORD_ARRAY);
	glTexCoordPointer(2, GL_FLOAT, 0, &tex0[0]);

	//TEX1
	for (int i = 0; i < tris; i++){
		tex1.push_back(0.0f); tex1.push_back(refrUV - move); tex1.push_back(0.0f); tex1.push_back(0.0f - move); tex1.push_back(refrUV); tex1.push_back(refrUV - move);
		tex1.push_back(refrUV); tex1.push_back(0.0f - move); tex1.push_back(refrUV); tex1.push_back(refrUV - move); tex1.push_back(0.0f); tex1.push_back(0.0f - move);
	}
	glClientActiveTexture(GL_TEXTURE1_ARB);
	glEnableClientState(GL_TEXTURE_COORD_ARRAY);
	glTexCoordPointer(2, GL_FLOAT, 0, &tex1[0]);

	//TEX2
	for (int i = 0; i < tris; i++){
		tex2.push_back(0.0f); tex2.push_back(normalUV + move2); tex2.push_back(0.0f); tex2.push_back(0.0f + move2); tex2.push_back(normalUV); tex2.push_back(0.0f + move2);
		tex2.push_back(normalUV); tex2.push_back(normalUV + move2); tex2.push_back(normalUV); tex2.push_back(0.0f + move2); tex2.push_back(0.0f); tex2.push_back(0.0f + move2);
	}
	glClientActiveTexture(GL_TEXTURE2_ARB);
	glEnableClientState(GL_TEXTURE_COORD_ARRAY);
	glTexCoordPointer(2, GL_FLOAT, 0, &tex2[0]);

	//TEX3
	for (int i = 0; i < tris; i++){
		tex3.push_back(0.0f); tex3.push_back(0.0f); tex3.push_back(0.0f); tex3.push_back(0.0f); tex3.push_back(0.0f); tex3.push_back(0.0f);
		tex3.push_back(0.0f); tex3.push_back(0.0f); tex3.push_back(0.0f); tex3.push_back(0.0f); tex3.push_back(0.0f); tex3.push_back(0.0f);
	}
	glClientActiveTexture(GL_TEXTURE3_ARB);
	glEnableClientState(GL_TEXTURE_COORD_ARRAY);
	glTexCoordPointer(2, GL_FLOAT, 0, &tex3[0]);

	//TEX4
	for (int i = 0; i < tris; i++){
		tex4.push_back(0.0f); tex4.push_back(0.0f); tex4.push_back(0.0f); tex4.push_back(0.0f); tex4.push_back(0.0f); tex4.push_back(0.0f);
		tex4.push_back(0.0f); tex4.push_back(0.0f); tex4.push_back(0.0f); tex4.push_back(0.0f); tex4.push_back(0.0f); tex4.push_back(0.0f);
	}
	glClientActiveTexture(GL_TEXTURE4_ARB);
	glEnableClientState(GL_TEXTURE_COORD_ARRAY);
	glTexCoordPointer(2, GL_FLOAT, 0, &tex4[0]);

	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(3, GL_FLOAT, 0, &waterGeom2[0]);
	glDrawArrays(GL_PATCHES, 0, 2647);

	glDisableClientState(GL_VERTEX_ARRAY);

	glDisableClientState(GL_TEXTURE_COORD_ARRAY);
	rp.tts.off();
}
