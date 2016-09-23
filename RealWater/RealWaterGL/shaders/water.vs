varying vec4 refrCoords; 
varying vec4 normCoords; 
varying vec4 viewCoords;
varying vec4 viewTangentSpace;
varying vec4 lightTangentSpace;

uniform vec4 lightPos_0; 
uniform vec4 cameraPos;

varying vec3 WorldPos_CS_in; 

void main()
{
	vec4 tangent = vec4(1.0, 0.0, 0.0, 0.0);
	vec4 normal = vec4(0.0, 1.0, 0.0, 0.0);
	vec4 biTangent = vec4(0.0, 0.0, 1.0, 0.0);
	
	//Compute View Tangent Space
	vec4 viewDir = cameraPos - gl_Vertex;
	viewTangentSpace.x = dot(viewDir, tangent);
	viewTangentSpace.y = dot(viewDir, biTangent);
	viewTangentSpace.z = dot(viewDir, normal);
	viewTangentSpace.w = 1.0;

	//Compute Light Tangent Space
	vec4 lightDir = lightPos_0 - gl_Vertex;
	lightTangentSpace.x = dot(lightDir, tangent);
	lightTangentSpace.y = dot(lightDir, biTangent);
	lightTangentSpace.z = dot(lightDir, normal);
	lightTangentSpace.w = 1.0;

	refrCoords = gl_MultiTexCoord1;
	normCoords = gl_MultiTexCoord2;
	
	viewCoords = gl_ModelViewProjectionMatrix * gl_Vertex;
	//WorldPos_CS_in = gl_Vertex.xyz;
	
	gl_Position = viewCoords;
}