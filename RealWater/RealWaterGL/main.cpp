#include<stdlib.h>
#include"Camera.h"
#include"Render.h"

Render _render;

void render(){
	_render.renderScene();
}

void animate(){
	_render.animateScene();
}

void resize_window(int w, int h){
	_render.resize(w, h);
}

int main(int argc, char **argv){
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowPosition(0, 0);
	glutInitWindowSize(WINDOW_SIZEX, WINDOW_SIZEY);
	glutCreateWindow("Lake Scene");

#if 1
	_render.initializeGL();
	_render.loadScene();
	//glutFullScreen();
	glutDisplayFunc(render);
	glutReshapeFunc(resize_window);
	glutIdleFunc(animate);
	glutMainLoop();

#else	

	HeightMap hm(1024,1024);
	hm.initialize();
	hm.generateWaves(0, 0);
#endif
}