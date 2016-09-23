#ifndef TEXTURE_H
#define TEXTURE_H

#include"Types.h"

class Texture
{
public:
	Texture();
	~Texture();
	int loadTextureFromFile(std::string filename); // load texture from <filename>.bmp
	void renderTexture(int size, int channels, int type);
	UINT getTextureBinding() { return this->tid; } //return binding to corresponding texture
private:
	UINT tid = NULL;
};

#endif

