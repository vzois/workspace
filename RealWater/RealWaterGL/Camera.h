#ifndef CAMERA_H
#define CAMERA_H

#include "Types.h"

#define SPEED 0.1f
#define MOUSE_SENSITIVITY 500.0f
#define Y_THRESHOLD 1.0f

class Camera
{
public:
	Camera();
	~Camera();
	void look();
	void position(Vector3 pos, Vector3 view, Vector3 up);

	Vector3 getPos() { return this->_pos; }
	Vector3 getView() { return this->_view; }
	Vector3 getUp() { return this->_up; }

	void checkMovement();
	void move(GLfloat step); // move forward (positive step) // move backwards (negative step)
	void strafe(GLfloat step);
	void mouseChangeView();
	void rotate(GLfloat angle, GLfloat x, GLfloat y, GLfloat z);
	void collisionAvoidance(GLfloat height);
	bool update();

	void frameRate();

private:
	Vector3 _up;
	Vector3 _pos;
	Vector3 _view;
	Vector3 _strafe;

	GLdouble deltaTime = 0.0f;
	GLdouble lastFrame = 0.0f;

	bool tessellation = false;
};

#endif
