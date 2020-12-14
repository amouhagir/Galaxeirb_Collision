#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>


//#include "model.h"
#define FILENAME "../dubinski.tab"
#define NB_PARTICULES 81920
#define  MASSFACTOR 10
#define  DAMP 1
#define  N 1024
#define DT 0.1
#define MODULO 80-1
#define NTHREAD 4
#define R1 1.0f
#define G1 1.0f
#define B1 1.0f
#define R2 1.0f
#define G2 1.0f
#define B2 0.0f


typedef struct particule 
{
	float m;
	float x;
	float y;
	float z;
	float vx;
	float vy;
	float vz;

} particule_t;

typedef struct vectors
{
	float x;
	float y;
	float z;

} vector_t;
