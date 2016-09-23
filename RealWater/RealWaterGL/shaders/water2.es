#version 410 compatibility

layout(triangles, equal_spacing, ccw) in;

uniform vec4 lightPos_0;
uniform vec4 cameraPos;
uniform sampler2D normalMap;
uniform sampler2D dispMap;                                                                          

in vec2 refrCoordsES[]; 
in vec2 normCoordsES[];
in vec3 posCoordsES[];

out vec2 refrCoordsFS; 
out vec2 normCoordsFS;
out vec4 posCoordsFS;

in vec3 viewTangentSpaceES[];
in vec3 lightTangentSpaceES[];

out vec4 viewTangentSpaceFS;
out vec4 lightTangentSpaceFS; 

out vec4 viewTangentSpace;
out vec4 lightTangentSpace; 

vec2 interpolate2D(vec2 v0, vec2 v1, vec2 v2)                                                   
{                                                                                               
    return vec2(gl_TessCoord.x) * v0 + vec2(gl_TessCoord.y) * v1 + vec2(gl_TessCoord.z) * v2;   
}                                                                                               

vec3 interpolate3D(vec3 v0, vec3 v1, vec3 v2)                                                   
{                                                                                               
    return vec3(gl_TessCoord.x) * v0 + vec3(gl_TessCoord.y) * v1 + vec3(gl_TessCoord.z) * v2;   
}                                                                                               

void main()
{
	vec4 vertPos = vec4(interpolate3D(posCoordsES[0], posCoordsES[1], posCoordsES[2]),1.0);
	
	viewTangentSpaceFS = vec4(interpolate3D(viewTangentSpaceES[0], viewTangentSpaceES[1], viewTangentSpaceES[2]),1.0);
	lightTangentSpaceFS = vec4(interpolate3D(lightTangentSpaceES[0], lightTangentSpaceES[1], lightTangentSpaceES[2]),1.0);

	//Interpolate Normals and Textures//
	refrCoordsFS = interpolate2D(refrCoordsES[0],refrCoordsES[1],refrCoordsES[2]);
	normCoordsFS = interpolate2D(normCoordsES[0],normCoordsES[1],normCoordsES[2]);
	
	posCoordsFS = gl_ModelViewProjectionMatrix * vertPos;
	//float dp = texture(dispMap,refrCoordsFS).x;
	float dp = texture(dispMap,vertPos.xy).x;
	vec4 newPos = vec4(vertPos.x,vertPos.y + dp*4, vertPos.z, 1.0);
	gl_Position = gl_ModelViewProjectionMatrix * newPos;
    //gl_Position = posCoordsFS;
}                                                                                               
