varying vec4 refrCoords; 
varying vec4 normCoords; 
varying vec4 viewCoords;
varying vec4 viewTangentSpace;
varying vec4 lightTangentSpace;

uniform sampler2D reflectTex;
uniform sampler2D refractTex;
uniform sampler2D normalMap;
uniform sampler2D dudvMap;
uniform sampler2D depthMap;
uniform vec4 waterColor;

varying vec3 WorldPos_FS_in;

#define GOLDEN_RATIO 1.61803398875
#define EULER_CONSTANT 2.718281828459
#define PI 3.14159265359

void main()
{
	const float shine = 32.0;
	const float dist = 0.025;
	const float refrac = 0.009;

	vec4 distOffset = texture2D(dudvMap, normCoords.xy) * dist;
	vec4 dudvColor = texture2D(dudvMap, vec2(refrCoords + distOffset));
	dudvColor = normalize(dudvColor * 2.0 - 1.0) * refrac;

	vec4 normalVector = texture2D(normalMap, vec2(refrCoords + distOffset));
	normalVector = normalVector * 2.0 - 1.0;
	normalVector.a = 0.0;

	//-lightTangentSpace because the positive vector is towards the light source not the surface ( look at vertex shader)
	vec4 lightReflection = normalize( reflect(-lightTangentSpace, normalVector) );
	float bias = 0.15;
	float scale = 0.009;
	float power = EULER_CONSTANT;
	float F = min(1.0,scale + scale*pow(1.0 + dot(normalVector, lightReflection),power));
	
	vec4 iF = vec4(F);
	vec4 fT = 1.0 - iF;

	vec4 projCoord = viewCoords / viewCoords.q;
	projCoord = (projCoord + 1.0) * 0.5;
	projCoord += dudvColor;
	projCoord = clamp(projCoord, 0.001, 0.999);
	
	vec4 reflectColor  = texture2D(reflectTex, projCoord.xy);
	vec4 refractColor  = texture2D(refractTex, projCoord.xy);
	vec4 depthValue = texture2D(depthMap, projCoord.xy);
		
	vec4 invDepth = 1.0 - depthValue;
	refractColor *= iF * invDepth;
	refractColor +=  waterColor * depthValue * iF;

	reflectColor *= fT;

	vec4 localView = normalize(viewTangentSpace);		
	float intensity = max(0.0, dot(lightReflection, localView) );
	vec4 specColor = vec4(pow(intensity, shine));

	gl_FragColor = refractColor + reflectColor + specColor;
}