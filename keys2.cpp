/*
 * Copyright 2011-2012 Noah Snavely, Cornell University
 * (snavely@cs.cornell.edu).  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:

 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above
 *    copyright notice, this list of conditions and the following
 *    disclaimer in the documentation and/or other materials provided
 *    with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY NOAH SNAVELY ''AS IS'' AND ANY EXPRESS
 * OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NOAH SNAVELY OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
 * OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
 * USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
 * DAMAGE.
 *
 * The views and conclusions contained in the software and
 * documentation are those of the authors and should not be
 * interpreted as representing official policies, either expressed or
 * implied, of Cornell University.
 *
 */

/* keys2.cpp */
/* Class for SIFT keypoints */

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <fstream>
#include <fcntl.h>
#include <sys/stat.h>
#include <stdint.h>
#include <iostream>
#include "keys2.h"

using namespace std;


enum
{
	SIFT_NAME = ('S' + ('I' << 8) + ('F' << 16) + ('T' << 24)),
	MSER_NAME = ('M' + ('S' << 8) + ('E' << 16) + ('R' << 24)),
	RECT_NAME = ('R' + ('E' << 8) + ('C' << 16) + ('T' << 24)),
	SIFT_VERSION_4 = ('V' + ('4' << 8) + ('.' << 16) + ('0' << 24)),
	SIFT_EOF = (0xff + ('E' << 8) + ('O' << 16) + ('F' << 24)),
};
const int SIFTBINS = 128;
int writeVisualSFMSiftGPU(const char* fn, float *KeyPts, unsigned char *desc, int nsift)
{
	int sift_name = SIFT_NAME, sift_version = SIFT_VERSION_4, KeyDim = 5, DescDim = 128, sift_eof = SIFT_EOF;
	ofstream fout;
	fout.open(fn, ios::binary);
	if (!fout.is_open())
	{
		cout << "Cannot write: " << fn << endl;
		return 1;
	}
	fout.write(reinterpret_cast<char *>(&sift_name), sizeof(int));
	fout.write(reinterpret_cast<char *>(&sift_version), sizeof(int));
	fout.write(reinterpret_cast<char *>(&nsift), sizeof(int));
	fout.write(reinterpret_cast<char *>(&KeyDim), sizeof(int));
	fout.write(reinterpret_cast<char *>(&DescDim), sizeof(int));
	for (int j = 0; j < nsift; ++j)
	{
		float x = KeyPts[4 * j], y = KeyPts[4 * j + 1], dummy = 0.f;
		fout.write(reinterpret_cast<char *>(&x), sizeof(float));
		fout.write(reinterpret_cast<char *>(&y), sizeof(float));
		fout.write(reinterpret_cast<char *>(&dummy), sizeof(float));
		fout.write(reinterpret_cast<char *>(&KeyPts[4 * j + 2]), sizeof(float));
		fout.write(reinterpret_cast<char *>(&KeyPts[4 * j + 3]), sizeof(float));
	}

	for (int j = 0; j < nsift; ++j)
		for (int i = 0; i < DescDim; i++)
			fout.write(reinterpret_cast<char *>(&desc[j * 128 + i]), sizeof(unsigned char));

	fout.write(reinterpret_cast<char *>(&sift_eof), sizeof(int));
	fout.close();

	return 0;
}

//This reads a keypoint file from a given filename and returns the list of keypoints. Currently it reads from visualsfm siftgpu output
int GetNumberOfKeys(const char *filename)
{
	int num;

	ifstream fin;
	fin.open(filename, ios::binary);
	if (!fin.is_open())
	{
		cout << "Cannot open: " << filename << endl;
		return 0;
	}

	int dummy;

	fin.read(reinterpret_cast<char *>(&dummy), sizeof(int));//SIFT
	fin.read(reinterpret_cast<char *>(&dummy), sizeof(int));///V4.0
	fin.read(reinterpret_cast<char *>(&num), sizeof(int));//npts
	fin.read(reinterpret_cast<char *>(&dummy), sizeof(int));//5 numbers
	fin.read(reinterpret_cast<char *>(&dummy), sizeof(int));//descriptorSize

	//Minh: hack
	int new_num = 0;
	float x, y, val, scale, ori;
	vector<bool>valid; valid.reserve(20000);
	for (int ii = 0; ii < num; ii++)
	{
		fin.read(reinterpret_cast<char *>(&x), sizeof(float));
		fin.read(reinterpret_cast<char *>(&y), sizeof(float));
		fin.read(reinterpret_cast<char *>(&val), sizeof(float));
		fin.read(reinterpret_cast<char *>(&scale), sizeof(float));
		fin.read(reinterpret_cast<char *>(&ori), sizeof(float));

		if (y < 300 || y>1920 - 300)
			valid.push_back(1), new_num++;
		else
			valid.push_back(0);
	}
	num = new_num;

	fin.close();

	return num;
}
int ReadKeyFile(const char *filename, short int **keys, keypt_t **info)
{
	ifstream fin;
	fin.open(filename, ios::binary);
	if (!fin.is_open())
	{
		cout << "Cannot open: " << filename << endl;
		return 0;
	}

	int dummy, npts, descriptorSize;
	float val;

	fin.read(reinterpret_cast<char *>(&dummy), sizeof(int));//SIFT
	fin.read(reinterpret_cast<char *>(&dummy), sizeof(int));///V4.0
	fin.read(reinterpret_cast<char *>(&npts), sizeof(int));//npts
	fin.read(reinterpret_cast<char *>(&dummy), sizeof(int));//5 numbers
	fin.read(reinterpret_cast<char *>(&descriptorSize), sizeof(int));//descriptorSize

	//Minh: hack
	int orgnpts = npts; npts = 0;
	float x, y, scale, ori;
	vector<bool>valid; valid.reserve(20000);
	vector<keypt_t> tkey; tkey.reserve(20000);
	for (int ii = 0; ii < orgnpts; ii++)
	{
		fin.read(reinterpret_cast<char *>(&x), sizeof(float));
		fin.read(reinterpret_cast<char *>(&y), sizeof(float));
		fin.read(reinterpret_cast<char *>(&val), sizeof(float));
		fin.read(reinterpret_cast<char *>(&scale), sizeof(float));
		fin.read(reinterpret_cast<char *>(&ori), sizeof(float));

		keypt_t tk; tk.x = x - 0.5, tk.y = y - 0.5, tk.scale = scale, tk.orient = ori;
		if (y < 300 || y>1920 - 300)
			valid.push_back(1), tkey.push_back(tk), npts++;
		else
			valid.push_back(0);
	}

	*keys = new short int[descriptorSize * npts];
	if (info != NULL)//info is not always requested
		*info = new keypt_t[npts];

	for (int ii = 0; ii < npts; ii++)
		if (info != NULL)
			(*info)[ii].x = tkey[ii].x, (*info)[ii].y = tkey[ii].y, (*info)[ii].scale = tkey[ii].scale, (*info)[ii].orient = tkey[ii].orient;

	uint8_t d;
	short int *p = *keys;
	short int *pt = new short int[descriptorSize]; int count = 0;
	for (int j = 0; j < orgnpts; j++)
	{
		for (int i = 0; i < descriptorSize; i++)
		{
			fin.read(reinterpret_cast<char *>(&d), sizeof(uint8_t));
			pt[i] = (short int)d;
		}
		if (valid[j])
		{
			for (int i = 0; i < descriptorSize; i++)
				p[count*descriptorSize + i] = pt[i];
			count++;
		}
	}
	delete[]pt;
	fin.close();

	/**keys = new short int[descriptorSize * npts];
	if (info != NULL)//info is not always requested
	*info = new keypt_t[npts];

	for (int ii = 0; ii < npts; ii++)
	{
	fin.read(reinterpret_cast<char *>(&x), sizeof(float));
	fin.read(reinterpret_cast<char *>(&y), sizeof(float));
	fin.read(reinterpret_cast<char *>(&val), sizeof(float));
	fin.read(reinterpret_cast<char *>(&scale), sizeof(float));
	fin.read(reinterpret_cast<char *>(&ori), sizeof(float));
	if (info != NULL)
	(*info)[ii].x = x - 0.5, (*info)[ii].y = y - 0.5, (*info)[ii].scale = scale, (*info)[ii].orient = ori;
	}


	uint8_t d;
	short int *p = *keys;
	for (int j = 0; j < npts; j++)
	{
	val = 0.0;
	for (int i = 0; i < descriptorSize; i++)
	{
	fin.read(reinterpret_cast<char *>(&d), sizeof(uint8_t));
	p[j*descriptorSize + i] = (short int)d;
	}
	}
	fin.close();*/

	return npts;
}

