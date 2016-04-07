/* keys2.h */
/* Class for SIFT keypoints */

#ifndef __keys2_h__
#define __keys2_h__

#include <vector>
#include <stdio.h>

//#include "zlib.h"


class Keypoint {
public:
	Keypoint(float x, float y, float scale, float ori, short int *d) :m_x(x), m_y(y), m_scale(scale), m_ori(ori), m_d(d){ }

	float m_x, m_y;        /* Subpixel location of keypoint. */
	float m_scale, m_ori;  /* Scale and orientation (range [-PI,PI]) */
	short int *m_d;    /* Vector of descriptor values */
};


/* Data struct for matches */
class KeypointMatch {
public:
	KeypointMatch()	{ }
	KeypointMatch(int idx1, int idx2) : m_idx1(idx1), m_idx2(idx2)	{ }
	int m_idx1, m_idx2;
};

typedef struct {
	float x, y;
	float scale;
	float orient;
} keypt_t;

/* Returns the number of keys in a file */
int GetNumberOfKeys(const char *filename);

/* This reads a keypoint file from a given filename and returns the list of keypoints. */
int ReadKeyFile(const char *filename, short int **keys, keypt_t **info = NULL);

int ReadKeyPositions(const char *filename, keypt_t **info);

/* Read keypoints from the given file pointer and return the list of
 * keypoints.  The file format starts with 2 integers giving the total
 * number of keypoints and the size of descriptor vector for each
 * keypoint (currently assumed to be 128). Then each keypoint is
 * specified by 4 floating point numbers giving subpixel row and
 * column location, scale, and orientation (in radians from -PI to
 * PI).  Then the descriptor vector for each keypoint is given as a
 * list of integers in range [0,255]. */
int ReadKeys(FILE *fp, short int **keys, keypt_t **info = NULL);

/* Read keys using MMAP to speed things up */
std::vector<Keypoint *> ReadKeysMMAP(FILE *fp);

int WriteBinaryKeyFile(const char *filename, int num_keys, const short int *keys, const keypt_t *info);

#endif /* __keys2_h__ */
