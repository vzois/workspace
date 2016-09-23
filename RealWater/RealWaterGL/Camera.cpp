#include "Camera.h"

Camera::Camera()
{
	this->_pos = Vector3(0.0f, 0.0f, 15.0f);
	this->_view = Vector3(0.0f, 1.0f, 0.0f);
	this->_up = Vector3(0.0f, 1.0f, 0.0f);
}


Camera::~Camera()
{
}

void Camera::position(Vector3 pos, Vector3 view, Vector3 up ){
	this->_pos = pos;
	this->_view = view;
	this->_up = up;
}

void Camera::look(){
	gluLookAt(
		this->_pos[X], this->_pos[Y], this->_pos[Z],
		this->_view[X], this->_view[Y], this->_view[Z],
		this->_up[X], this->_up[Y], this->_up[Z]
		);
}

void Camera::mouseChangeView(){
	POINT mPos;									
	int mX = WINDOW_SIZEX >> 1;				
	int mY = WINDOW_SIZEY >> 1;				
	float angleY = 0.0f;							
	float angleZ = 0.0f;							
	static float rotX = 0.0f;
	static float lrotX = 0.0f;

	GetCursorPos(&mPos);
	if ((mPos.x == mX) && (mPos.y == mY)) return;
	SetCursorPos(mX, mY);

	angleY = (float)((mX - mPos.x)) / MOUSE_SENSITIVITY;
	angleZ = (float)((mY - mPos.y)) / MOUSE_SENSITIVITY;
	lrotX = rotX;
	rotX += angleZ;

	if (rotX > Y_THRESHOLD){
		rotX = Y_THRESHOLD;
		if (lrotX != Y_THRESHOLD && false){
			Vector3 yAxis = cross(this->_view - this->_pos, this->_up);
			yAxis = normalize(yAxis);
			this->rotate(Y_THRESHOLD - lrotX, yAxis[X], yAxis[Y], yAxis[Z]);
		}
	}
	else if (rotX < -Y_THRESHOLD){
		rotX = -Y_THRESHOLD;
		if (lrotX != -Y_THRESHOLD && false){
			Vector3 yAxis = cross(this->_view - this->_pos, this->_up);
			yAxis = normalize(yAxis);
			this->rotate(-Y_THRESHOLD - lrotX, yAxis[X], yAxis[Y], yAxis[Z]);
		}
	}
	else
	{
		Vector3 yAxis = cross(this->_view - this->_pos, this->_up);
		yAxis = normalize(yAxis);
		this->rotate(angleZ, yAxis[X], yAxis[Y], yAxis[Z]);
	}

	this->rotate(angleY, 0, 1, 0);
}

void Camera::rotate(GLfloat angle, GLfloat x, GLfloat y, GLfloat z){
	Vector3 vNewView;
	Vector3 vView = this->_view - this->_pos;

	GLfloat ctheta = (GLfloat)cos(angle);
	GLfloat stheta = (GLfloat)sin(angle);

	vNewView.v[X] = (ctheta + (1 - ctheta) * x * x)		* vView.v[X];
	vNewView.v[X] += ((1 - ctheta) * x * y - z * stheta)	* vView.v[Y];
	vNewView.v[X] += ((1 - ctheta) * x * z + y * stheta)	* vView.v[Z];

	vNewView.v[Y] = ((1 - ctheta) * x * y + z * stheta)	* vView.v[X];
	vNewView.v[Y] += (ctheta + (1 - ctheta) * y * y)		* vView.v[Y];
	vNewView.v[Y] += ((1 - ctheta) * y * z - x * stheta)	* vView.v[Z];

	vNewView.v[Z] = ((1 - ctheta) * x * z - y * stheta)	* vView.v[X];
	vNewView.v[Z] += ((1 - ctheta) * y * z + x * stheta)	* vView.v[Y];
	vNewView.v[Z] += (ctheta + (1 - ctheta) * z * z)		* vView.v[Z];

	this->_view = this->_pos + vNewView;
}

void Camera::checkMovement(){

	float speed = SPEED*this->deltaTime;
	static bool wireFrame = false;
	static int keyPress = 0;

	if (GetKeyState(VK_UP) & 0x80 || GetKeyState('W') & 0x80) move(speed);// MOVE FORWARD
	if (GetKeyState(VK_DOWN) & 0x80 || GetKeyState('S') & 0x80) move(-speed); // MOVE BACKWARD
	if (GetKeyState(VK_LEFT) & 0x80 || GetKeyState('A') & 0x80) strafe(-speed);// STRAFE LEFT
	if (GetKeyState(VK_RIGHT) & 0x80 || GetKeyState('D') & 0x80) strafe(speed);// STRAFE RIGHT
	if (GetKeyState(VK_ESCAPE) & 0x80) PostQuitMessage(0);//QUIT PROGRAM
	
	
	if (GetKeyState('X') & 0x80 && keyPress > 20){
		wireFrame = !wireFrame;
		if (wireFrame) glPolygonMode(GL_FRONT, GL_LINE);
		else glPolygonMode(GL_FRONT, GL_FILL);
		keyPress = 0;
	}

	if (GetKeyState('Z') & 0x80 && keyPress > 20){ 
		tessellation = !tessellation;
		keyPress = 0;
	}
	keyPress++;

}

void Camera::move(GLfloat step){
	Vector3 cView = this->_view - this->_pos;
	cView = normalize(cView);
	cView = cView * step;

	this->_pos = this->_pos + cView;
	this->_view = this->_view + cView;
}

void Camera::strafe(GLfloat step){
	Vector3 ss = this->_strafe*step;

	this->_pos.v[X] = this->_pos[X] + ss[X];
	this->_pos.v[Z] = this->_pos[Z] + ss[Z];

	this->_view.v[X] = this->_view[X] + ss[X];
	this->_view.v[Z] = this->_view[Z] + ss[Z];
}

void Camera::collisionAvoidance(GLfloat height){
	Vector3 newPos = this->getPos();

	if (this->getPos()[Y] < height + 10)
	{
		newPos.v[Y] = (GLfloat)height+ 10;
		GLfloat diff = newPos[Y] - this->getPos()[Y];
		Vector3 newView = this->getView();
		newView.v[Y] += diff;
		this->position(newPos, newView, Vector3(0, 1, 0));
	}
}

bool Camera::update(){
	Vector3 _cstrafe = cross(this->_view - this->_pos, this->_up);
	this->_strafe = normalize(_cstrafe);

	this->mouseChangeView();
	this->checkMovement();
	this->frameRate();

	return tessellation;
}

void Camera::frameRate(){
	GLdouble currFrame = GetTickCount();
	deltaTime = currFrame - lastFrame;
	lastFrame = currFrame;
}