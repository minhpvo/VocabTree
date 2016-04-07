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


#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <ctime>
#include <assert.h>
#include <string>
#include <vector>
#include <map>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <stdint.h>
#include <algorithm>
#include "ThirdParty/ann_1.1/include/ANN/ANN.h"
#include "ThirdParty/ann_1.1_char/include/ANN/ANN.h"
#include "defines.h"
#include "qsort.h"
#include "VocabTree.h"
#include "keys2.h"

#define MAX_ARRAY_SIZE 8388608 // 2 ** 23

using namespace std;
using namespace ann_1_1_char;

unsigned char *ReadKeys(const char *keyfile, int dim, int &num_keys_out)
{
	short int *keys;
	keypt_t *info = NULL;
	int num_keys = ReadKeyFile(keyfile, &keys, &info);

	unsigned char *keys_char = new unsigned char[num_keys * dim];
	for (int j = 0; j < num_keys * dim; j++)
		keys_char[j] = (unsigned char)keys[j];

	delete[] keys;
	if (info != NULL)
		delete[] info;

	num_keys_out = num_keys;
	return keys_char;
}
// const double min_feature_scale = 0.0; // 1.4; // 1.2
unsigned char *ReadAndFilterKeys(const char *keyfile, int dim, double min_feature_scale, int max_keys, int &num_keys_out)
{
	short int *keys;
	keypt_t *info = NULL;
	int num_keys = ReadKeyFile(keyfile, &keys, &info);
	if (num_keys == 0)
	{
		num_keys_out = 0;
		return NULL;
	}

	//Filter keys
	int num_keys_filtered = 0;
	unsigned char *keys_char = new unsigned char[num_keys * dim];
	if (min_feature_scale == 0.0 && max_keys == 0)
	{
		for (int j = 0; j < num_keys * dim; j++)
			keys_char[j] = (unsigned char)keys[j];
		num_keys_filtered = num_keys;
	}
	else
	{
		for (int j = 0; j < num_keys; j++)
		{
			if (info[j].scale < min_feature_scale)
				continue;

			for (int k = 0; k < dim; k++)
				keys_char[num_keys_filtered * dim + k] = (unsigned char)keys[j * dim + k];

			num_keys_filtered++;
			if (max_keys > 0 && num_keys_filtered >= max_keys)
				break;
		}
	}

	delete[] keys;
	if (info != NULL)
		delete[] info;

	num_keys_out = num_keys_filtered;

	return keys_char;
}
unsigned char *ReadDescriptorFile(const char *desc_file, unsigned int dim, int &num_keys_out)
{
	ifstream fin;
	fin.open(desc_file, ios::binary);
	if (!fin.is_open())
	{
		cout << "Cannot open: " << desc_file << endl;
		return NULL;
	}

	int dummy, npts, descriptorSize;
	float val;

	fin.read(reinterpret_cast<char *>(&dummy), sizeof(int));//SIFT
	fin.read(reinterpret_cast<char *>(&dummy), sizeof(int));///V4.0
	fin.read(reinterpret_cast<char *>(&npts), sizeof(int));//npts
	fin.read(reinterpret_cast<char *>(&dummy), sizeof(int));//5 numbers
	fin.read(reinterpret_cast<char *>(&descriptorSize), sizeof(int));//descriptorSize

	float x, y, scale, ori;
	for (int ii = 0; ii < npts; ii++)
	{
		fin.read(reinterpret_cast<char *>(&x), sizeof(float));
		fin.read(reinterpret_cast<char *>(&y), sizeof(float));
		fin.read(reinterpret_cast<char *>(&val), sizeof(float));
		fin.read(reinterpret_cast<char *>(&scale), sizeof(float));
		fin.read(reinterpret_cast<char *>(&ori), sizeof(float));
	}


	uint8_t d;
	unsigned char *desc = new unsigned char[npts * dim];
	for (int j = 0; j < npts; j++)
	{
		for (int i = 0; i < descriptorSize; i++)
		{
			fin.read(reinterpret_cast<char *>(&d), sizeof(uint8_t));
			desc[j*descriptorSize + i] = d;
		}
	}
	fin.close();

	num_keys_out = npts;
	return desc;
}

//learn a vocabulary tree through hierarchical kmeans 
int VocabLearn(const char *list_in, int depth, int bf, int restarts, const char *tree_out)
{
	printf("Building tree with depth: %d, branching factor: %d, ""and restarts: %d\n", depth, bf, restarts);
	FILE *f = fopen(list_in, "r");
	if (f == NULL) {
		printf("Could not open file: %s\n", list_in);
		return 1;
	}

	std::vector<std::string> key_files;
	char buf[256];
	while (fgets(buf, 256, f))
	{
		if (buf[strlen(buf) - 1] == '\n')
			buf[strlen(buf) - 1] = 0; // Remove trailing newline

		key_files.push_back(std::string(buf));
	}
	fclose(f);

	int num_files = (int)key_files.size();
	unsigned long total_keys = 0;
	for (int i = 0; i < num_files; i++) {
		int num_keys = GetNumberOfKeys(key_files[i].c_str());
		total_keys += num_keys;
	}
	printf("Total number of keys: %lu\n", total_keys);

	// Reduce the branching factor if need be if there are not enough keys, to avoid problems later.
	if (bf >= (int)total_keys){
		bf = total_keys - 1;
		printf("Reducing the branching factor to: %d\n", bf);
	}

	fflush(stdout);

	int dim = 128;
	unsigned long long len = (unsigned long long) total_keys * dim;
	int num_arrays = len / MAX_ARRAY_SIZE + ((len % MAX_ARRAY_SIZE) == 0 ? 0 : 1);

	printf("Allocating %llu bytes in total, in %d arrays\n", len, num_arrays);
	unsigned long long total = 0;
	unsigned char **vs = new unsigned char *[num_arrays];
	for (int i = 0; i < num_arrays; i++)
	{
		unsigned long long remainder = len - total;
		unsigned long len_curr = MIN(remainder, MAX_ARRAY_SIZE);

		printf("Allocating array of size %lu\n", len_curr); fflush(stdout);
		vs[i] = new unsigned char[len_curr];

		remainder -= len_curr;
	}

	// Create the array of pointers 
	printf("Allocating pointer array of size %lu\n", 4 * total_keys); fflush(stdout);
	unsigned char **vp = new unsigned char *[total_keys];

	int curr_array = 0;
	unsigned long off = 0, curr_key = 0;
	for (int i = 0; i < num_files; i++)
	{
		printf("  Reading keyfile %s\n", key_files[i].c_str()); fflush(stdout);

		short int *keys;
		int num_keys = ReadKeyFile(key_files[i].c_str(), &keys);
		if (num_keys > 0)
		{
			for (int j = 0; j < num_keys; j++)
			{
				for (int k = 0; k < dim; k++)
					vs[curr_array][off + k] = keys[j * dim + k];

				vp[curr_key] = vs[curr_array] + off;
				curr_key++;
				off += dim;
				if (off == MAX_ARRAY_SIZE)
				{
					off = 0;
					curr_array++;
				}
			}
		}
		delete[] keys;
	}

	VocabTree tree;
	tree.Build(total_keys, dim, depth, bf, restarts, vp);
	tree.Write(tree_out);

	return 0;
}
// Building a database with a vocabulary tree
int VocabDB(char *list_in, char *tree_in, char *db_out, int use_tfidf = 1, int normalize = 1, int start_id = 0, DistanceType distance_type = DistanceMin)
{
	double min_feature_scale = 1.4;

	switch (distance_type) {
	case DistanceDot:
		printf("[VocabMatch] Using distance Dot\n");
		break;
	case DistanceMin:
		printf("[VocabMatch] Using distance Min\n");
		break;
	default:
		printf("[VocabMatch] Using no known distance!\n");
		break;
	}

	FILE *f = fopen(list_in, "r");
	if (f == NULL) {
		printf("Error opening file %s for reading\n", list_in);
		return 1;
	}

	std::vector<std::string> key_files;
	char buf[256];
	while (fgets(buf, 256, f))
	{
		/* Remove trailing newline */
		if (buf[strlen(buf) - 1] == '\n')
			buf[strlen(buf) - 1] = 0;

		key_files.push_back(std::string(buf));
	}

	printf("[VocabBuildDB] Reading tree %s...\n", tree_in);
	fflush(stdout);

	VocabTree tree;
	tree.Read(tree_in);
	tree.Flatten();
	tree.m_distance_type = distance_type;
	tree.SetInteriorNodeWeight(0.0);

	//Initialize leaf weights to 1.0 
	tree.SetConstantLeafWeights();

	const int dim = 128;
	int num_db_images = (int)key_files.size();
	unsigned long count = 0;

	tree.ClearDatabase();

	for (int i = 0; i < num_db_images; i++)
	{
		int num_keys = 0;
		unsigned char *keys = ReadAndFilterKeys(key_files[i].c_str(), dim, min_feature_scale, 0, num_keys);

		printf("[VocabBuildDB] Adding vector %d (%d keys)\n", start_id + i, num_keys);
		tree.AddImageToDatabase(start_id + i, num_keys, keys);

		if (num_keys > 0)
			delete[] keys;
	}

	printf("[VocabBuildDB] Pushed %lu features\n", count);
	fflush(stdout);

	if (use_tfidf == 1)
		tree.ComputeTFIDFWeights(num_db_images);

	if (normalize == 1)
		tree.NormalizeDatabase(start_id, num_db_images);

	printf("[VocabBuildDB] Writing database ...\n");
	tree.Write(db_out);

	// char filename[256];
	// sprintf(filename, "vectors_%03d.txt", start_id);
	// tree.WriteDatabaseVectors(filename, start_id, num_db_images);

	return 0;
}

//Read a database stored as a vocab tree and score a set of query images 
int VocabMatch(char *db_in, char *list_in, char *query_in, int num_nbrs, char *matches_out, DistanceType distance_type = DistanceMin, int normalize = 1)
{
	const int dim = 128;

	printf("[VocabMatch] Using database %s\n", db_in);
	switch (distance_type)
	{
	case DistanceDot:
		printf("[VocabMatch] Using distance Dot\n");
		break;
	case DistanceMin:
		printf("[VocabMatch] Using distance Min\n");
		break;
	default:
		printf("[VocabMatch] Using no known distance!\n");
		break;
	}

	// Read the tree 
	printf("[VocabMatch] Reading database...\n"); fflush(stdout);
	clock_t start = clock();
	VocabTree tree;
	tree.Read(db_in);
	clock_t end = clock();
	printf("[VocabMatch] Read database in %0.3fs\n", (double)(end - start) / CLOCKS_PER_SEC);

	tree.Flatten();
	tree.SetDistanceType(distance_type);
	tree.SetInteriorNodeWeight(0, 0.0);

	// Read the database keyfiles 
	FILE *f = fopen(list_in, "r");
	if (f == NULL)
	{
		printf("Could not open file: %s\n", list_in);
		return 1;
	}

	std::vector<std::string> db_files;
	char buf[256];
	while (fgets(buf, 256, f))
	{
		// Remove trailing newline 
		if (buf[strlen(buf) - 1] == '\n')
			buf[strlen(buf) - 1] = 0;

		db_files.push_back(std::string(buf));
	}
	fclose(f);

	//Read the query keyfiles
	f = fopen(query_in, "r");
	if (f == NULL)
	{
		printf("Could not open file: %s\n", query_in);
		return 1;
	}

	std::vector<std::string> query_files;
	while (fgets(buf, 256, f))
	{
		// Remove trailing newline 
		if (buf[strlen(buf) - 1] == '\n')
			buf[strlen(buf) - 1] = 0;

		char keyfile[256]; sscanf(buf, "%s", keyfile);
		query_files.push_back(std::string(keyfile));
	}
	fclose(f);

	int num_db_images = (int)db_files.size();
	int num_query_images = (int)query_files.size();
	printf("[VocabMatch] Read %d database images\n", num_db_images);

	//Now score each query keyfile
	printf("[VocabMatch] Scoring %d query images...\n", num_query_images); fflush(stdout);

	float *scores = new float[num_db_images];
	double *scores_d = new double[num_db_images];
	int *perm = new int[num_db_images];

	FILE *f_match = fopen(matches_out, "w");
	if (f_match == NULL)
	{
		printf("[VocabMatch] Error opening file %s for writing\n", matches_out);
		return 1;
	}

	bool bnormalize = normalize == 1 ? true : false;
	for (int i = 0; i < num_query_images; i++)
	{
		start = clock();

		for (int j = 0; j < num_db_images; j++)
			scores[j] = 0.0;

		int num_keys;
		unsigned char *keys = ReadKeys(query_files[i].c_str(), dim, num_keys);

		clock_t start_score = clock();
		double mag = tree.ScoreQueryKeys(num_keys, bnormalize, keys, scores);
		clock_t end_score = end = clock();

		printf("[VocabMatch] Scored image %s in %0.3fs ( %0.3fs total, num_keys = %d, mag = %0.3f )\n", query_files[i].c_str(), (double)(end_score - start_score) / CLOCKS_PER_SEC, (double)(end - start) / CLOCKS_PER_SEC, num_keys, mag);

		//Find the top scores 
		for (int j = 0; j < num_db_images; j++)
			scores_d[j] = (double)scores[j];

		qsort_descending();
		qsort_perm(num_db_images, scores_d, perm);

		int top = MIN(num_nbrs, num_db_images);

		for (int j = 0; j < top; j++)
		{
			// if (perm[j] == index_i)
			//     continue;
			fprintf(f_match, "%d %d %0.4f\n", i, perm[j], scores_d[j]);
			//fprintf(f_match, "%d %d %0.4f\n", i, perm[j], mag - scores_d[j]);
		}
		fflush(f_match); fflush(stdout);
		delete[] keys;
	}
	fclose(f_match);

	delete[] scores, delete[] scores_d, delete[] perm;
	return 0;
}

// combining several vocab trees into one
int combineVocab(vector<string> &allTreesIn, char *tree_out)
{
	int num_trees = (int)allTreesIn.size();

	VocabTree tree;
	tree.Read(allTreesIn[0].c_str());

	//Start with the second tree, as we just read the first one
	for (int i = 1; i < num_trees; i++)
	{
		printf("[VocabCombine] Adding tree %d [%s]...\n", i, allTreesIn[i].c_str());
		fflush(stdout);

		VocabTree tree_add;
		tree_add.Read(allTreesIn[i].c_str());
		tree.Combine(tree_add);
		tree_add.Clear();
	}

	//Now do the reweighting:
	int total_num_db_images = tree.GetMaxDatabaseImageIndex() + 1;
	printf("Total num_db_images: %d\n", total_num_db_images);
	tree.ComputeTFIDFWeights(total_num_db_images);
	tree.NormalizeDatabase(0, total_num_db_images);
	tree.Write(tree_out);

	//Write vectors to a file 
	tree.WriteDatabaseVectors("vectors_all.txt", 0, total_num_db_images);

	return 0;
}
// compare images feature using db
int VocabCompare(int argc, char **argv)
{
	if (argc != 5 && argc != 6)
	{
		printf("Usage: %s <tree.in> <image1.key> <image2.key> <matches.out> [distance_type]\n", argv[0]);
		return 1;
	}

	char *tree_in = argv[1];
	char *image1_in = argv[2];
	char *image2_in = argv[3];
	char *matches_out = argv[4];
	DistanceType distance_type = DistanceMin;

	if (argc >= 6)
		distance_type = (DistanceType)atoi(argv[5]);

	switch (distance_type) {
	case DistanceDot:
		printf("[VocabMatch] Using distance Dot\n");
		break;
	case DistanceMin:
		printf("[VocabMatch] Using distance Min\n");
		break;
	default:
		printf("[VocabMatch] Using no known distance!\n");
		break;
	}

	printf("[VocabBuildDB] Reading tree %s...\n", tree_in); fflush(stdout);
	VocabTree tree;
	tree.Read(tree_in);

	printf("[VocabCompare] Flattening tree...\n");
	tree.Flatten();
	tree.m_distance_type = distance_type;
	tree.SetInteriorNodeWeight(0.0);

	/* Initialize leaf weights to 1.0 */
	tree.SetConstantLeafWeights();

	const int dim = 128;

	tree.ClearDatabase();

	int num_keys_1 = 0, num_keys_2 = 0;
	unsigned char *keys1 = ReadKeys(image1_in, dim, num_keys_1);
	unsigned char *keys2 = ReadKeys(image2_in, dim, num_keys_2);

	unsigned long *ids1 = new unsigned long[num_keys_1];
	unsigned long *ids2 = new unsigned long[num_keys_2];

	printf("[VocabCompare] Adding image 0 (%d keys)\n", num_keys_1);
	tree.AddImageToDatabase(0, num_keys_1, keys1, ids1);
	if (num_keys_1 > 0)
		delete[] keys1;

	printf("[VocabCompare] Adding image 1 (%d keys)\n", num_keys_2);
	tree.AddImageToDatabase(1, num_keys_2, keys2, ids2);
	if (num_keys_2 > 0)
		delete[] keys2;

	// tree.ComputeTFIDFWeights();
	tree.NormalizeDatabase(0, 2);

	//Find collisions among visual word IDs 
	std::multimap<unsigned long, unsigned int> word_map;
	for (unsigned int i = 0; i < (unsigned int)num_keys_1; i++)
	{
		printf("0 %d -> %lu\n", i, ids1[i]);
		std::pair<unsigned long, unsigned int> elem(ids1[i], i);
		word_map.insert(elem);
	}

	//Count number of matches 
	int num_matches = 0;
	for (unsigned int i = 0; i < (unsigned int)num_keys_2; i++)
	{
		unsigned long id = ids2[i];
		printf("1 %d -> %lu\n", i, ids2[i]);

		std::pair<std::multimap<unsigned long, unsigned int>::iterator, std::multimap<unsigned long, unsigned int>::iterator> ret;
		ret = word_map.equal_range(id);

		unsigned int size = 0;
		std::multimap<unsigned long, unsigned int>::iterator iter;
		for (iter = ret.first; iter != ret.second; iter++)
			size++, num_matches++;

		if (size > 0)
			printf("size[%lu] = %d\n", id, size);
	}
	printf("number of matches: %d\n", num_matches); fflush(stdout);

	FILE *f = fopen(matches_out, "w");
	if (f == NULL)
	{
		printf("Error opening file %s for writing\n", matches_out);
		return 1;
	}
	fprintf(f, "%d\n", num_matches);

	/* Write matches */
	for (unsigned int i = 0; i < (unsigned int)num_keys_2; i++)
	{
		unsigned long id = ids2[i];

		std::pair<std::multimap<unsigned long, unsigned int>::iterator, std::multimap<unsigned long, unsigned int>::iterator> ret;
		ret = word_map.equal_range(id);

		std::multimap<unsigned long, unsigned int>::iterator iter;
		for (iter = ret.first; iter != ret.second; iter++)
			fprintf(f, "%d %d\n", iter->second, i);
	}
	fclose(f);

	// printf("[VocabBuildDB] Writing tree...\n");
	// tree.Write(tree_out);

	return 0;
}

typedef struct {
	int id1, id2;
} matchID;

int GenerateUnduplicatedCorpusMatchesList(char *Path, int knnMax)
{
	char Fname[512];

	//image index must start from 0
	sprintf(Fname, "%s/Corpus/matches.txt", Path);	FILE *fp = fopen(Fname, "r");
	if (fp == NULL)
	{
		printf("Cannot load %s\n", Fname);
		return 1;
	}

	float s;
	int id1, id2, newid = -1, knnCount = 0, nimages = 0;

	matchID mid;
	vector<matchID > matches;  matches.reserve(1e6);
	while (fscanf(fp, "%d %d %f", &mid.id1, &mid.id2, &s) != EOF)
	{
		if (newid == -1 || newid != mid.id1)
			newid = mid.id1, knnCount = 0;
		if (knnCount < knnMax)
			matches.push_back(mid), knnCount++;
		nimages = max(nimages, mid.id1);
	}
	fclose(fp);
	nimages++;

	bool  *validMatches = new bool[nimages*nimages];
	for (int ii = 0; ii < nimages*nimages; ii++)
		validMatches[ii] = false;
	for (int ii = 0; ii < (int)matches.size(); ii++)
	{
		validMatches[matches[ii].id1 + nimages*matches[ii].id2] = true;
		validMatches[matches[ii].id2 + nimages*matches[ii].id1] = true;
	}

	///ccreate matching list
	sprintf(Fname, "%s/Corpus/vsfmPairs.txt", Path); fp = fopen(Fname, "w+");
	for (int ii = 0; ii < nimages; ii++)
	{
		for (int jj = ii + 1; jj < nimages; jj++)
		{
			if (validMatches[ii + jj*nimages])
				fprintf(fp, "%s/Corpus/%d.jpg %s/Corpus/%d.jpg\n", Path, ii, Path, jj);
		}
	}
	fclose(fp);

	delete[]validMatches;
	return 0;
}
int GenerateCameraIMatchList(char *Path, int CameraI, int startF, int stopF, int increF, int knnMax)
{
	char Fname[512];

	sprintf(Fname, "%s/%d/VocabMatches.txt", Path, CameraI);	FILE *fp = fopen(Fname, "r");
	if (fp == NULL)
	{
		printf("Cannot load %s\n", Fname);
		return 1;
	}

	float s;
	int id1, id2, newid = -1, knnCount = 0, nimages = 0;

	matchID mid;
	vector<matchID > matches;  matches.reserve(1e6);
	while (fscanf(fp, "%d %d %f", &mid.id1, &mid.id2, &s) != EOF)
	{
		if (newid == -1 || newid != mid.id1)
			newid = mid.id1, knnCount = 0;
		if (knnCount < knnMax)
			matches.push_back(mid), knnCount++;
		nimages = max(nimages, mid.id1);
	}
	fclose(fp);
	nimages++;


	sprintf(Fname, "%s/%d/ToMatch2.txt", Path, CameraI); fp = fopen(Fname, "w");
	vector<int> tID;
	for (int fid = startF; fid <= stopF; fid += increF)
	{
		tID.clear();
		for (int jj = 0; jj < (int)matches.size(); jj++)
		{
			if (matches[jj].id1 == fid / increF) //matches id1 is the order of the file in the querry list
				tID.push_back(matches[jj].id2); // the order of the corpus image should be the same as its filename
		}
		if (tID.size() > 0)
		{
			fprintf(fp, "%d %d ", fid, (int)tID.size());
			for (int jj = 0; jj < tID.size(); jj++)
				fprintf(fp, "%d ", tID[jj]);
			fprintf(fp, "\n");
		}
	}
	fclose(fp);

	return 0;
}

int main(int argc, char **argv)
{
	if (argc == 1)
	{
		printf("Usage:\n");
		printf("Learn VocabTree:\ %s <mode:0> <list.in> <depth> <branching_factor> " "<restarts> <tree.out>\n", argv[0]);
		printf("Genereate VocabHistDB: %s <mode: 1> <list.in> <tree.in> <db.out> [use_tfidf:1] [normalize:1] [start_id:0] [distance_type:1]\n", argv[0]);
		printf("Match to VocabHistDB: %s <mode:2> <db.in> <list.in> <query.in> <num_nbrs> <matches.out> [distance_type:1] [normalize:1]\n", argv[0]);
		printf("Combine VocabTree: %s <mode:3> <tree1.in> <tree2.in> ... <tree.out>\n", argv[0]);
		printf("Group corpus via VocabHistDB: %s <mode:4> <MaxImgToMatch>\n", argv[0]);
		printf("Find corpus Img to match: %s <mode:5> <CamID> <startF> <stopF> <increF> <MaxImgToMatch>\n", argv[0]);
		return 1;
	}

	int mode = atoi(argv[1]);
	if (mode == 0)
	{
		if (argc != 7)
		{
			printf("Usage: %s <mode:0> <list.in> <depth> <branching_factor> " "<restarts> <tree.out>\n", argv[0]);
			return 1;
		}

		const char *list_in = argv[2];
		int depth = atoi(argv[3]), bf = atoi(argv[4]), restarts = atoi(argv[5]);
		const char *tree_out = argv[6];

		//const char *list_in = "E:/Traffic/Corpus/list.txt";
		//int depth = 0, bf = 10000, restarts = 1;
		//const char *tree_out = "E:/Traffic/Corpus/vocabTree.txt";

		VocabLearn(list_in, depth, bf, restarts, tree_out);

		return 0;
	}
	else if (mode == 1)
	{
		if (argc < 5 || argc > 9)
		{
			printf("Usage: %s <mode: 1> <list.in> <tree.in> <db.out> [use_tfidf:1] [normalize:1] [start_id:0] [distance_type:1]\n", argv[0]);
			return 1;
		}

		char *list_in = argv[2], *tree_in = argv[3], *db_out = argv[4];
		//char *list_in = "E:/Traffic/Corpus/list.txt", *tree_in = "E:/Traffic/Corpus/vocabTree.txt", *db_out = "E:/Traffic/Corpus/vocabdb.txt";

		int use_tfidf = 1, normalize = 1, start_id = 0;
		DistanceType distance_type = DistanceMin;
		if (argc >= 6)
			use_tfidf = atoi(argv[5]);
		if (argc >= 7)
			normalize = atoi(argv[6]);
		if (argc >= 8)
			start_id = atoi(argv[7]);
		if (argc >= 9)
			distance_type = (DistanceType)atoi(argv[8]);

		VocabDB(list_in, tree_in, db_out, use_tfidf, normalize, start_id, distance_type);

		return 0;
	}
	else if (mode == 2)
	{
		if (argc != 7 && argc != 8 && argc != 9)
		{
			printf("Usage: %s <mode:2> <db.in> <list.in> <query.in> <num_nbrs> <matches.out> [distance_type:1] [normalize:1]\n", argv[0]);
			return 1;
		}
		char *db_in = argv[2], *list_in = argv[3], *query_in = argv[4];
		int num_nbrs = atoi(argv[5]);
		char *matches_out = argv[6];

		/*char *db_in = "E:/Traffic/Corpus/vocabdb.txt",
			*list_in = "E:/Traffic/Corpus/list.txt",
			*query_in = "E:/Traffic/Corpus/list.txt";
			int num_nbrs = 100;
			char *matches_out = "E:/Traffic/Corpus/matches.txt";*/

		DistanceType distance_type = DistanceMin;
		int normalize = 1;
		if (argc >= 8)
			distance_type = (DistanceType)atoi(argv[7]);
		if (argc >= 9)
			normalize = (atoi(argv[8]) != 0);

		VocabMatch(db_in, list_in, query_in, num_nbrs, matches_out, distance_type, normalize);

		return 0;
	}
	else if (mode == 3)
	{
		if (argc < 4)
		{
			printf("Usage: %s <mode:3> <tree1.in> <tree2.in> ... <tree.out>\n", argv[0]);
			return 1;
		}

		int num_trees = argc - 3;
		char *tree_out = argv[argc - 1];

		vector<string> allTrees;
		for (int ii = 0; ii < num_trees; ii++)
			allTrees.push_back(argv[ii + 2]);

		combineVocab(allTrees, tree_out);
		return 0;
	}
	else if (mode == 4)
	{
		if (argc != 4)
		{
			printf("Usage: %s <mode:4> <MaxImgToMatch>\n", argv[0]);
			return 1;
		}
		char *Path = argv[2];
		int knnMax = atoi(argv[3]);

		GenerateUnduplicatedCorpusMatchesList(Path, knnMax);

		return 0;
	}
	else if (mode == 5)
	{
		if (argc != 8)
		{
			printf("Usage: %s <mode:5> <CamID> <startF> <stopF> <increF> <MaxImgToMatch>\n", argv[0]);
			return 1;
		}
		char *Path = argv[2];
		int camID = atoi(argv[3]), startF = atoi(argv[4]), stopF = atoi(argv[5]), increF = atoi(argv[6]), knnMax = atoi(argv[7]);

		GenerateCameraIMatchList(Path, camID, startF, stopF, increF, knnMax);
		return 0;
	}

	return 0;
}
