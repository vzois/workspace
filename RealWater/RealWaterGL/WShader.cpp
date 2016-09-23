#include "WShader.h"

WShader::WShader()
{
}

WShader::~WShader()
{
}

void WShader::initShader(bool tess){
	char *glExt = (char*)glGetString(GL_EXTENSIONS);
	std::string vs = tess ? VERTEX_PROC_1 : VERTEX_PROC_0;
	std::string cs = tess ? TESS_CONTROL_1 : "";
	std::string es = tess ? TESS_EVAL_1 : "";
	std::string fs = tess ? FRAGMENT_PROC_1 : FRAGMENT_PROC_0;

	if (!strstr(glExt, "GL_ARB_shader_objects")) handleError("Error while initializing Shader = GL_ARB_shader_objects extension not supported!");
	if (!strstr(glExt, "GL_ARB_shading_language_100")) handleError("Error while initializing Shader = GL_ARB_shading_language_100 extension not supported!");

	this->program = glCreateProgram();
	this->vertexProc(vs);
	if (tess){
		this->tessallationControl(cs);
		this->tessellationEvaluation(es);
	}
	this->fragmentProc(fs);
	this->finalize();
}

void WShader::vertexProc(std::string vertex_processor){
	std::string source = this->loadProcCode(vertex_processor);
	const char *src = source.c_str();
	this->vertexShader = glCreateShader(GL_VERTEX_SHADER_ARB);

	std::cout << "vertex shader: " << vertex_processor << std::endl;
	glShaderSource(this->vertexShader, 1, &src, NULL);
	glCompileShader(this->vertexShader);
	glAttachShader(this->program, this->vertexShader);
}

void WShader::fragmentProc(std::string fragment_processor){
	std::string source = this->loadProcCode(fragment_processor);
	const char *src = source.c_str();
	this->fragmentShader = glCreateShader(GL_FRAGMENT_SHADER_ARB);

	std::cout << "fragment shader: " << fragment_processor << std::endl;
	glShaderSource(this->fragmentShader, 1, &src, NULL);
	glCompileShader(this->fragmentShader);
	glAttachShader(this->program, this->fragmentShader);
}

void  WShader::tessallationControl(std::string tess_control){
	std::string source = this->loadProcCode(tess_control);
	const char *src = source.c_str();
	this->tessellationControlShader = glCreateShader(GL_TESSELATION_CONTROL_SHADER_ARB);

	std::cout << "tesselation control: " << tess_control << std::endl;
	glShaderSource(this->fragmentShader, 1, &src, NULL);
	glShaderSource(this->tessellationControlShader, 1, &src, NULL);
	glCompileShader(this->tessellationControlShader);
	glAttachShader(this->program, this->tessellationControlShader);
}

void  WShader::tessellationEvaluation(std::string tess_evaluation){
	std::string source = this->loadProcCode(tess_evaluation);
	const char *src = source.c_str();
	this->tessellationEvaluationShader = glCreateShader(GL_TESSELATION_EVALUATION_SHADER_ARB);

	std::cout << "tesselation evaluation: " << tess_evaluation << std::endl;
	glShaderSource(this->tessellationEvaluationShader, 1, &src, NULL);
	glCompileShader(this->tessellationEvaluationShader);
	glAttachShader(this->program, this->tessellationEvaluationShader);
}

bool WShader::finalize(){
	GLint Success = 0;
	GLchar ErrorLog[1024] = { 0 };

	glLinkProgram(this->program);

	glGetProgramiv(this->program, GL_LINK_STATUS, &Success);
	if (Success == 0) {
		glGetProgramInfoLog(this->program, sizeof(ErrorLog), NULL, ErrorLog);
		fprintf(stderr, "Error linking shader program: '%s'\n", ErrorLog);
		return false;
	}

	glValidateProgram(this->program);
	glGetProgramiv(this->program, GL_VALIDATE_STATUS, &Success);
	if (!Success) {
		glGetProgramInfoLog(this->program, sizeof(ErrorLog), NULL, ErrorLog);
		fprintf(stderr, "Invalid shader program: '%s'\n", ErrorLog);
		return false;
	}

	return glGetError() == GL_NO_ERROR;
}

std::string WShader::loadProcCode(std::string path){
	std::ifstream fs(path.c_str());
	std::string source="";
	std::string line="";

	if (!fs)  handleError("Error reading source = "+path);

	while (getline(fs,line)){ source = source + "\n" + line; }
	fs.close();

	return source;
}