#ifndef WSHADER_H
#define WSHADER_H

#include "Types.h"

#define APIENTRYP APIENTRY *

#define GL_VERTEX_SHADER_ARB              0x8B31
#define GL_FRAGMENT_SHADER_ARB            0x8B30
#define GL_TESSELATION_CONTROL_SHADER_ARB  0x8E88
#define GL_TESSELATION_EVALUATION_SHADER_ARB  0x8E87

typedef GLuint GLShaderProgram;

#include<fstream>
#define VERTEX_PROC_0 "shaders/water.vs"
#define FRAGMENT_PROC_0 "shaders/water.fs"

#define VERTEX_PROC_1 "shaders/water2.vs"
#define FRAGMENT_PROC_1 "shaders/water2.fs"
#define TESS_CONTROL_1 "shaders/water2.cs"
#define TESS_EVAL_1 "shaders/water2.es"

#include <iostream>

class WShader
{
public:
	WShader();
	~WShader();

	void initShader(bool tess);
	void vertexProc(std::string vertex_processor);
	void fragmentProc(std::string fragment_processor);
	void tessallationControl(std::string tess_control);
	void tessellationEvaluation(std::string tess_evaluation);
	bool finalize();

	std::string loadProcCode(std::string path);

	GLint getLocation(std::string var){ return glGetUniformLocation(this->getProgram(), var.c_str()); }
	void setVal(GLint var, GLint val){ glUniform1i(var, val); }
	void setVector(GLint var, Vector3 vv, GLfloat a){ glUniform4f(var, vv[X] , vv[Y], vv[Z],a); }
	void setMatrix(GLint var, Matrix4 m){ glUniformMatrix4fv(var, 1, GL_TRUE, (const GLfloat*)m.m); }

	void on(){ glUseProgram(this->program); }
	void off(){ glUseProgram(0); }

	GLShaderProgram getProgram(){ return this->program; }
private:
	GLShaderProgram vertexShader;
	GLShaderProgram fragmentShader;
	GLShaderProgram tessellationControlShader;
	GLShaderProgram tessellationEvaluationShader;

	GLShaderProgram program;
};

#endif
