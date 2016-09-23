#version 410 compatibility

uniform vec4 lightPos_0;
uniform vec4 cameraPos;
uniform sampler2D dispMap;  

out vec2 refrCoordsCS; 
out vec2 normCoordsCS; 
out vec3 posCoordsCS;

out vec3 viewTangentSpaceCS;
out vec3 lightTangentSpaceCS;                                                                        

void main()                                                                                     
{
	vec4 tex = texture2D(dispMap,gl_Vertex.xy);
	vec4 tangent = vec4(1.0, 0.0, 0.0, 0.0);
	vec4 normal = vec4(0.0, 1.0, 0.0, 0.0);
	vec4 biTangent = vec4(0.0, 0.0, 1.0, 0.0);
	
	//Compute View Tangent Space
	vec4 viewDir = cameraPos - gl_Vertex;
	viewTangentSpaceCS.x = dot(viewDir, tangent);
	viewTangentSpaceCS.y = dot(viewDir, biTangent);
	viewTangentSpaceCS.z = dot(viewDir, normal);

	//Compute Light Tangent Space
	vec4 lightDir = lightPos_0 - gl_Vertex;
	lightTangentSpaceCS.x = dot(lightDir, tangent);
	lightTangentSpaceCS.y = dot(lightDir, biTangent);
	lightTangentSpaceCS.z = dot(lightDir, normal);

	posCoordsCS = gl_Vertex.xyz;
	refrCoordsCS = vec2(gl_MultiTexCoord1.xy);
	normCoordsCS = vec2(gl_MultiTexCoord2.xy);
}
