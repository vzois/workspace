#ifndef TYPES_H
#define TYPES_H

#include <GL\glew.h>
#include <GL\glut.h>
#include <GL\glu.h>	
#include <string>
#include <math.h>
#include <time.h>

#define PI 3.14159265359

#define SUCCESS 0
#define ERROR 1

#define WINDOW_SIZEX 1024
#define WINDOW_SIZEY 768

#define WINDOW_POSX 0
#define WINDOW_POSY 0

#define X 0
#define Y 1
#define Z 2
#define W 3

#define U 0
#define V 1

#define DEFAULT_FOV 45.0

#define DEFAULT_NEAR 0.9f
#define DEFAULT_FAR 4000.0f

#define MAX_TEXTURES 200
#define MAX_HEIGHTMAPS 100

#define FRAME_DELAY 1.0f

#define BOX_TEX_PATH "textures/skybox/"

/*Skybox Texture Flags*/
#define BOX_FRONT 0
#define BOX_BACK 1
#define BOX_TOP 2
#define BOX_BOTTOM 3
#define BOX_LEFT 4
#define BOX_RIGHT 5

//TERRAIN TEXTURES AND HEIGHMAPS
#define TERRAIN_TEX_PATH "textures/Terrain.bmp"
#define TERRAIN_TEX 6
#define DETAIL_TERRAIN_TEX_PATH "textures/Detail2.bmp"
#define DETAIL_TERRAIN_TEX 7
#define DETAIL_SCALE 8 /*Scale of Detail for Second Texture*/

#define TERRAIN_HMAP_PATH "heightmaps/Terrain.raw"
#define TERRAIN_HMAP 0		/*Static Heightmap Flag*/
#define TERRAIN_SIZE 1024	/*Height Map Size*/
#define TERRAIN_STEP 16		/*Increase vertex count*/
#define TERRAIN_RATIO 1 /*Increase Height-Map values by RATIO*/

//PROCEDURAL TERRAIN HEIGHTMAP
#define PROC_TERRAIN_01 1
#define PROC_TERRAIN_02 2
#define PROC_TERRAIN_03 3
#define PROC_TERRAIN_04 4
#define PROC_TERRAIN_05 5

#define PROC_WAVE_01 6
#define PROC_WAVE_02 7
#define PROC_WAVE_03 8

#define WAVE_TEX_PATH "heightmaps/waves/"
#define WAVE_OFFSET 58
#define WAVE_HMAP_NUM 42

//WATER TEXTURE AND CHARACTERISTICS// WATER TEXTURE AVAILABLE FLAGS 16 - 25
#define WATER_HEIGHT 40.0f
#define WAVE_HEIGHT 32.0f
#define WATER_TEX_PATH "textures/water/Water.bmp"
#define WATER_TEX 16
#define NORMAL_MAP_TEX_PATH "textures/water/normalmap.bmp"
#define NORMAL_MAP_TEX 17
#define DUDV_MAP_TEX_PATH "textures/water/dudvmap.bmp"
#define DUDV_MAP_TEX 18

#define TEXTURE_SIZE 512
#define REFLECTION_TEX 19
#define REFRACTION_TEX 20
#define DEPTH_TEX 21

#define WATER_UV 35.0f 
#define WATER_FLOW 0.0015f
#define NORMAL_MAP_SCALE 0.25f

//WATER CAUSTIC TEXTURE // CAUSTIC TEXTURE AVAILABLE FLAGS 26 - 56
#define CAUSTIC_NUM 32
#define CAUSTIC_PATH "textures/water/caustics/"
#define CAUSTIC_OFFSET 26
#define CAUSTIC_SCALE 10.0f
#define CAUSTIC_ANIM 5

static GLfloat radian(GLfloat x){ return (GLfloat)(((x)* PI / 180.0f)); }

/*3D Vector STRUCT*/
struct Vector3{
public:
	Vector3(){}

	Vector3(GLfloat XX, GLfloat YY, GLfloat ZZ){
		v[X] = XX;
		v[Y] = YY;
		v[Z] = ZZ;
	}

	GLfloat operator[](int i){ return v[i]; }
	Vector3 operator+(Vector3 p){ return Vector3(v[X] + p[X], v[Y] + p[Y], v[Z] + p[Z]); }
	Vector3 operator-(Vector3 p){ return Vector3(v[X] - p[X], v[Y] - p[Y], v[Z] - p[Z]); }
	Vector3 operator*(GLfloat s){ return Vector3(v[X] * s, v[Y] * s, v[Z] * s); }
	Vector3 operator/(GLfloat s){ return Vector3(v[X] / s, v[Y] / s, v[Z] / s); }

	GLfloat v[3];
};

/*VECTOR OPERATIONS*/
static GLfloat magnitude(Vector3 vv){
	return (GLfloat)sqrt(vv[X] * vv[X] + vv[Y] * vv[Y] + vv[Z] * vv[Z]);
}

static Vector3 normalize(Vector3 vv){
	GLfloat denom = magnitude(vv);
	return vv / denom;
}

static Vector3 cross(Vector3 vv1, Vector3 vv2){
	return Vector3(
		(vv1[Y] * vv2[Z]) - (vv1[Z] * vv2[Y]),
		(vv1[Z] * vv2[X]) - (vv1[X] * vv2[Z]),
		(vv1[X] * vv2[Y]) - (vv1[Y] * vv2[X])
		);
}

struct Matrix4{

	Matrix4(){}

	inline Matrix4 operator*(const Matrix4& Right) const
	{
		Matrix4 Ret;

		for (unsigned int i = 0; i < 4; i++) {
			for (unsigned int j = 0; j < 4; j++) {
				Ret.m[i][j] = m[i][0] * Right.m[0][j] +
					m[i][1] * Right.m[1][j] +
					m[i][2] * Right.m[2][j] +
					m[i][3] * Right.m[3][j];
			}
		}

		return Ret;
	}

	void  perspProj(){
		const float ar = (float)WINDOW_SIZEX / WINDOW_SIZEY;
		const float zRange = DEFAULT_NEAR - DEFAULT_FAR;
		const float tanHalfFOV = tanf(radian(DEFAULT_FOV / 2.0f));

		m[0][0] = 1.0f / (tanHalfFOV * ar); m[0][1] = 0.0f;            m[0][2] = 0.0f;            m[0][3] = 0.0;
		m[1][0] = 0.0f;                   m[1][1] = 1.0f / tanHalfFOV; m[1][2] = 0.0f;            m[1][3] = 0.0;
		m[2][0] = 0.0f;                   m[2][1] = 0.0f;            m[2][2] = (-DEFAULT_NEAR - DEFAULT_FAR) / zRange; m[2][3] = 2.0f*DEFAULT_FAR*DEFAULT_NEAR / zRange;
		m[3][0] = 0.0f;                   m[3][1] = 0.0f;            m[3][2] = 1.0f;            m[3][3] = 0.0;
	}

	void translation(float x, float y, float z)
	{
		m[0][0] = 1.0f; m[0][1] = 0.0f; m[0][2] = 0.0f; m[0][3] = x;
		m[1][0] = 0.0f; m[1][1] = 1.0f; m[1][2] = 0.0f; m[1][3] = y;
		m[2][0] = 0.0f; m[2][1] = 0.0f; m[2][2] = 1.0f; m[2][3] = z;
		m[3][0] = 0.0f; m[3][1] = 0.0f; m[3][2] = 0.0f; m[3][3] = 1.0f;
	}

	void cameraTransform(const Vector3& Target, const Vector3& Up)
	{
		Vector3 n = normalize(Target);
		Vector3 u = normalize(Up); u = cross(u, n);
		Vector3 v = cross(n, u);

		m[0][0] = u[X];   m[0][1] = u[Y];   m[0][2] = u[Z];   m[0][3] = 0.0f;
		m[1][0] = v[X];   m[1][1] = v[Y];   m[1][2] = v[Z];   m[1][3] = 0.0f;
		m[2][0] = n[X];   m[2][1] = n[Y];   m[2][2] = n[Z];   m[2][3] = 0.0f;
		m[3][0] = 0.0f;  m[3][1] = 0.0f;  m[3][2] = 0.0f;  m[3][3] = 1.0f;
	}

	GLfloat m[4][4];
};

static Matrix4 getView(Vector3 cameraPos, Vector3 target, Vector3 up){
	Matrix4 cameraTT, cameraRT;

	cameraTT.translation(-cameraPos[X], -cameraPos[Y], -cameraPos[Z]);
	cameraRT.cameraTransform(target, up);

	return cameraRT * cameraTT;
}

static Matrix4 getVP(Vector3 camera, Vector3 target, Vector3 up){
	Matrix4 vt = getView(camera, target, up);
	Matrix4 pp;
	pp.perspProj();

	return pp * vt;
}

/*UTILITY FUNCTIONS*/
static bool frameRate(int desiredFrameRate)
{
	static float lastTime = GetTickCount() * 0.001f;
	static float elapsedTime = 0.0f;

	float currentTime = GetTickCount() * 0.001f;
	float deltaTime = currentTime - lastTime;
	float desiredFPS = 1.0f / desiredFrameRate;

	elapsedTime += deltaTime;
	lastTime = currentTime;

	if (elapsedTime > desiredFPS)
	{
		elapsedTime -= desiredFPS;
		return true;
	}

	return false;
}

/*ERROR HANDLER*/
static int handleError(std::string msg){
	MessageBox(NULL, msg.c_str(), "Error", MB_OK);
	return ERROR;
}


#endif