#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <sys/time.h>
#include <unistd.h>
#include "GL/glew.h"

#include "SDL2/SDL.h"
#include "SDL2/SDL_opengl.h"

#include "text.h"

#define FILENAME "dubinski.tab"
#define NB_PARTICULES 81920
#define  MASSFACTOR 10
#define  DAMP 1
#define  N 1024
#define DT 0.1
#define MODULO 80-1

static float g_inertia = 0.5f;

static float oldCamPos[] = { 0.0f, 0.0f, -45.0f };
static float oldCamRot[] = { 0.0f, 0.0f, 0.0f };
static float newCamPos[] = { 0.0f, 0.0f, -45.0f };
static float newCamRot[] = { 0.0f, 0.0f, 0.0f };

static bool g_showGrid = true;
static bool g_showAxes = true;

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



particule_t * Load_file(const char * filename, particule_t *pN){

	FILE* f =fopen(filename, "r");
	if ( f == NULL)
	{
		printf("Impossible to open the file\n");
	    	exit(1);
	}
	particule_t *p = (particule_t *) malloc(sizeof(particule_t) * NB_PARTICULES);
	int i,j=0;
	for (i = 0; i < NB_PARTICULES; ++i)
	{
		fscanf(f,"%f %f %f %f %f %f %f\n",&p[i].m,&p[i].x,&p[i].y,&p[i].z,&p[i].vx,&p[i].vy,&p[i].vz);
		if(i%MODULO == 0 && j<N){
			pN[j] = p[i];
			j++;
		}

	}

	fclose(f);
	return p;
}


void DrawGridXZ( float ox, float oy, float oz, int w, int h, float sz ) {

	int i;

	glLineWidth( 1.0f );

	glBegin( GL_LINES );

	glColor3f( 0.48f, 0.48f, 0.48f );

	for ( i = 0; i <= h; ++i ) {
		glVertex3f( ox, oy, oz + i * sz );
		glVertex3f( ox + w * sz, oy, oz + i * sz );
	}

	for ( i = 0; i <= h; ++i ) {
		glVertex3f( ox + i * sz, oy, oz );
		glVertex3f( ox + i * sz, oy, oz + h * sz );
	}

	glEnd();

}

void ShowAxes() {

	glLineWidth( 2.0f );

	glBegin( GL_LINES );

	glColor3f( 1.0f, 0.0f, 0.0f );
	glVertex3f( 0.0f, 0.0f, 0.0f );
	glVertex3f( 2.0f, 0.0f, 0.0f );

	glColor3f( 0.0f, 1.0f, 0.0f );
	glVertex3f( 0.0f, 0.0f, 0.0f );
	glVertex3f( 0.0f, 2.0f, 0.0f );

	glColor3f( 0.0f, 0.0f, 1.0f );
	glVertex3f( 0.0f, 0.0f, 0.0f );
	glVertex3f( 0.0f, 0.0f, 2.0f );

	glEnd();

}

void ShowPoint(float x, float y, float z, float r, float g, float b, float size) {

	glPointSize(size);

	glBegin( GL_POINTS );
	glColor3f( r, g, b );
	glVertex3f( x, y, z );

	glEnd();
}
void update_velocity(particule_t *p){
 	//double dt = 0.1;
	//int N=1024; //N° of particules in simulation
	int i,j,k=0;


	//Calcul d'accelaration
	float ax=0;
	float ay =0;
	float az =0;
	float dx,dy,dz,d,fact;

	for (j = 0; j < N; ++j)
	{
		ax=0;ay=0;az=0;
		for (i = 0; i < j; ++i)
		{
				dx = p[i].x-p[j].x;
				dy = p[i].y-p[j].y;
				dz = p[i].z-p[j].z;

				d  = dx*dx+dy*dy+dz*dz;  ///
				if ( d < 1.0 ) d = 1.0; ///
				fact=p[i].m/(d*sqrtf(d)); ///
				ax += dx*fact; ///
				ay += dy*fact;
				az += dz*fact;

		}
		for (i = j+1; i < N; ++i)
		{
				dx = p[i].x-p[j].x;
				dy = p[i].y-p[j].y;
				dz = p[i].z-p[j].z;

				d  = dx*dx+dy*dy+dz*dz;  ///
				if ( d < 1.0 ) d = 1.0; ///
				fact=p[i].m/(d*sqrtf(d)); ///
				ax += dx*fact; ///
				ay += dy*fact;
				az += dz*fact;

		}

		p[j].vx +=ax*MASSFACTOR*DAMP;
		p[j].vy +=ay*MASSFACTOR*DAMP;
		p[j].vz +=az*MASSFACTOR*DAMP;
	}


}

void Update_position(particule_t *p)
{
	int i=0;
	for (i = 0; i < N; ++i)
	{
		p[i].x += p[i].vx*DT;
		p[i].y += p[i].vy*DT;
		p[i].z += p[i].vz*DT;
	}
}


int main( int argc, char ** argv ) {

	SDL_Event event;
	SDL_Window * window;
	SDL_DisplayMode current;
	time_t t_Dload,t_Fload,t_Dveloc,t_Fveloc,t_Dpos,t_Fpos;

	particule_t *particules;
	particules = (particule_t*) malloc(sizeof(particule_t) * NB_PARTICULES);
	particule_t *part;
	part = (particule_t*) malloc(sizeof(particule_t) * N);

	int width = 640;
	int height = 480;

	bool done = false;

	float mouseOriginX = 0.0f;
	float mouseOriginY = 0.0f;

	float mouseMoveX = 0.0f;
	float mouseMoveY = 0.0f;

	float mouseDeltaX = 0.0f;
	float mouseDeltaY = 0.0f;

	struct timeval begin, end;
	float fps = 0.0;
	char sfps[40] = "FPS: ";

	if ( SDL_Init ( SDL_INIT_EVERYTHING ) < 0 ) {
		printf( "error: unable to init sdl\n" );
		return -1;
	}

	if ( SDL_GetDesktopDisplayMode( 0, &current ) ) {
		printf( "error: unable to get current display mode\n" );
		return -1;
	}

	window = SDL_CreateWindow( "SDL", 	SDL_WINDOWPOS_CENTERED,
										SDL_WINDOWPOS_CENTERED,
										width, height,
										SDL_WINDOW_OPENGL );

	SDL_GLContext glWindow = SDL_GL_CreateContext( window );

	GLenum status = glewInit();

	if ( status != GLEW_OK ) {
		printf( "error: unable to init glew\n" );
		return -1;
	}

	if ( ! InitTextRes( "./bin/DroidSans.ttf" ) ) {
		printf( "error: unable to init text resources\n" );
		return -1;
	}

	SDL_GL_SetSwapInterval( 1 );


	particules = Load_file(FILENAME, part);

	printf("%f %f %f\n", part[0].x,part[1].y,part[2].z);


	while ( !done ) {

		int i;

		while ( SDL_PollEvent( &event ) ) {

			unsigned int e = event.type;

			if ( e == SDL_MOUSEMOTION ) {
				mouseMoveX = event.motion.x;
				mouseMoveY = height - event.motion.y - 1;
			} else if ( e == SDL_KEYDOWN ) {
				if ( event.key.keysym.sym == SDLK_F1 ) {
					g_showGrid = !g_showGrid;
				} else if ( event.key.keysym.sym == SDLK_F2 ) {
					g_showAxes = !g_showAxes;
				} else if ( event.key.keysym.sym == SDLK_ESCAPE ) {
 					done = true;
				}
			}

			if ( e == SDL_QUIT ) {
				printf( "quit\n" );
				done = true;
			}

		}

		mouseDeltaX = mouseMoveX - mouseOriginX;
		mouseDeltaY = mouseMoveY - mouseOriginY;

		if ( SDL_GetMouseState( 0, 0 ) & SDL_BUTTON_LMASK ) {
			oldCamRot[ 0 ] += -mouseDeltaY / 5.0f;
			oldCamRot[ 1 ] += mouseDeltaX / 5.0f;
		}else if ( SDL_GetMouseState( 0, 0 ) & SDL_BUTTON_RMASK ) {
			oldCamPos[ 2 ] += ( mouseDeltaY / 100.0f ) * 0.5 * fabs( oldCamPos[ 2 ] );
			oldCamPos[ 2 ]  = oldCamPos[ 2 ] > -5.0f ? -5.0f : oldCamPos[ 2 ];
		}

		mouseOriginX = mouseMoveX;
		mouseOriginY = mouseMoveY;

		glViewport( 0, 0, width, height );
		glClearColor( 0.2f, 0.2f, 0.2f, 1.0f );
		glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
		glEnable( GL_BLEND );
		glBlendEquation( GL_FUNC_ADD );
		glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
		glDisable( GL_TEXTURE_2D );
		glEnable( GL_DEPTH_TEST );
		glMatrixMode( GL_PROJECTION );
		glLoadIdentity();
		gluPerspective( 50.0f, (float)width / (float)height, 0.1f, 100000.0f );
		glMatrixMode( GL_MODELVIEW );
		glLoadIdentity();

		for ( i = 0; i < 3; ++i ) {
			newCamPos[ i ] += ( oldCamPos[ i ] - newCamPos[ i ] ) * g_inertia;
			newCamRot[ i ] += ( oldCamRot[ i ] - newCamRot[ i ] ) * g_inertia;
		}

		glTranslatef( newCamPos[0], newCamPos[1], newCamPos[2] );
		glRotatef( newCamRot[0], 1.0f, 0.0f, 0.0f );
		glRotatef( newCamRot[1], 0.0f, 1.0f, 0.0f );

		if ( g_showGrid ) {
			DrawGridXZ( -100.0f, 0.0f, -100.0f, 20, 20, 10.0 );
		}

		if ( g_showAxes ) {
			ShowAxes();
		}

		gettimeofday( &begin, NULL );

		//Affichage des particules :
		int j=0;

		for (j = 0; j < N; ++j)
		{
			ShowPoint(part[j].x, part[j].y, part[j].z, 1.0f, 0.0f, 0.0f, 1.0f);
		}

		//Mise à jour des positions
		update_velocity(part);
	  Update_position(part);



		gettimeofday( &end, NULL );

		fps = (float)1.0f / ( ( end.tv_sec - begin.tv_sec ) * 1000000.0f + end.tv_usec - begin.tv_usec) * 1000000.0f;
		sprintf( sfps, "FPS : %.4f", fps );

		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		gluOrtho2D(0, width, 0, height);
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();

		DrawText( 10, height - 20, sfps, TEXT_ALIGN_LEFT, RGBA(255, 255, 255, 255) );
		DrawText( 10, 30, "'F1' : show/hide grid", TEXT_ALIGN_LEFT, RGBA(255, 255, 255, 255) );
		DrawText( 10, 10, "'F2' : show/hide axes", TEXT_ALIGN_LEFT, RGBA(255, 255, 255, 255) );

		SDL_GL_SwapWindow( window );
		SDL_UpdateWindowSurface( window );

	}

	SDL_GL_DeleteContext( glWindow );
	DestroyTextRes();
	SDL_DestroyWindow( window );
	SDL_Quit();

	return 1;
}
