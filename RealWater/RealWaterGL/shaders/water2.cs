#version 410 compatibility

layout (vertices = 3) out;

in vec2 refrCoordsCS[]; 
in vec2 normCoordsCS[];
in vec3 posCoordsCS[];

out vec2 refrCoordsES[]; 
out vec2 normCoordsES[]; 
out vec3 posCoordsES[];                                                                      

in vec3 viewTangentSpaceCS[];
in vec3 lightTangentSpaceCS[];

out vec3 viewTangentSpaceES[];
out vec3 lightTangentSpaceES[];

void main()                                                                                     
{   
	refrCoordsES[gl_InvocationID] = refrCoordsCS[gl_InvocationID];
	normCoordsES[gl_InvocationID] = normCoordsCS[gl_InvocationID];
    posCoordsES[gl_InvocationID] = posCoordsCS[gl_InvocationID];                          
	
	viewTangentSpaceES[gl_InvocationID] = viewTangentSpaceCS[gl_InvocationID];
	lightTangentSpaceES[gl_InvocationID] = lightTangentSpaceCS[gl_InvocationID];
	
	gl_TessLevelInner[0] = 16.0;
	gl_TessLevelOuter[0] = 16.0;
	gl_TessLevelOuter[1] = 16.0;
	gl_TessLevelOuter[2] = 16.0;
}     