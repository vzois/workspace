#include "Texture.h"

Texture::Texture()
{
}


Texture::~Texture()
{
}

int Texture::loadTextureFromFile(std::string filename){
	HBITMAP bmp_handler;
	BITMAP bmp;

	glGenTextures(1, &this->tid);
	bmp_handler = (HBITMAP)LoadImage(GetModuleHandle(NULL), filename.c_str(), IMAGE_BITMAP, 0, 0, LR_CREATEDIBSECTION | LR_LOADFROMFILE);
	if (bmp_handler == NULL) return handleError("Failed Loading Texture from ( "+filename+" )");
	GetObject(bmp_handler, sizeof(bmp), &bmp);

	glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
	glBindTexture(GL_TEXTURE_2D, this->tid);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, 3, bmp.bmWidth, bmp.bmHeight, 0, GL_BGR_EXT, GL_UNSIGNED_BYTE, bmp.bmBits);

	DeleteObject(bmp_handler);
	return SUCCESS;
}

void Texture::renderTexture(int size, int channels, int type){
	unsigned int *tex = NULL;
	tex = new unsigned int[size * size * channels];
	memset(tex, 0, size * size * channels * sizeof(unsigned int));

	glGenTextures(1, &this->tid);
	glBindTexture(GL_TEXTURE_2D, this->tid);

	glTexImage2D(GL_TEXTURE_2D, 0, channels, size, size, 0, type, GL_UNSIGNED_INT, tex);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	delete[] tex;
}

