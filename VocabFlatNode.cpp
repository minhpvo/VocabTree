/* VocabFlatNode.cpp */

#include "VocabTree.h"
#include "ThirdParty/ann_1.1_char/include/ANN/ANN.h"

using namespace ann_1_1_char;

#define NUM_NNS 1 // 3
#define SIGMA_SQ 6250.0
#define USE_SOFT_ASSIGNMENT 0

unsigned long VocabTreeFlatNode::PushAndScoreFeature(unsigned char *v, unsigned int index, int bf, int dim, bool add)
{
	int nn_idx[NUM_NNS];
	ann_1_1_char::ANNdist distsq[NUM_NNS];

	ann_1_1_char::annMaxPtsVisit(256);
	// annMaxPtsVisit(128);

	m_tree->annkPriSearch(v, NUM_NNS, nn_idx, distsq, 0.0);

	unsigned long r;
	if (USE_SOFT_ASSIGNMENT)
	{
		double w_weights[NUM_NNS];

		double sum = 0.0;
		for (int i = 0; i < NUM_NNS; i++)
		{
			w_weights[i] = exp(-(double)distsq[i] / (2.0 * SIGMA_SQ));
			sum += w_weights[i];
		}

		for (int i = 0; i < NUM_NNS; i++)
			w_weights[i] /= sum;

		for (int i = 0; i < NUM_NNS; i++)
		{
			// printf("dist: %0.3f, w_weight: %0.3f\n", (double) distsq[i], w_weight);
			double w_weight = w_weights[i];
			r = m_children[nn_idx[i]]->PushAndScoreFeature(v, index, bf, dim, add);
		}
	}
	else
		r = m_children[nn_idx[0]]->PushAndScoreFeature(v, index, bf, dim, add);

	return r;
}

/* Create a search tree for the given set of keypoints */
void VocabTreeFlatNode::BuildANNTree(int num_leaves, int dim)
{
	// unsigned long mem_size = num_leaves * dim;
	// unsigned char *desc = new unsigned char[mem_size];

	/* Create a new array of points */
	ann_1_1_char::ANNpointArray pts = ann_1_1_char::annAllocPts(num_leaves, dim);

	unsigned long id = 0;
	FillDescriptors(num_leaves, dim, id, pts[0]);

	/* Create a search tree for k2 */
	m_tree = new ann_1_1_char::ANNkd_tree(pts, num_leaves, dim, 16);
}
