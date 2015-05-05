#include "../Intersector.h"
#include "../Mixture.h"
#include <vector>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>


struct Detection : public FFLD::Rectangle
{
	FFLD::HOGPyramid::Scalar score;

	Detection() : score(0)
	{
	}

	Detection(FFLD::HOGPyramid::Scalar score, FFLD::Rectangle bndbox) : FFLD::Rectangle(bndbox), score(score)
	{
	}

	bool operator<(const Detection & detection) const
	{
		return score > detection.score;
	}
};

// Load mixture model from filepath
bool load_mixture_model(const std::string filepath, FFLD::Mixture& mixture);


// Detect using a loaded model
void detect(const FFLD::Mixture & mixture, const unsigned char* image,
            const int width, const int height, const int padding,
            const int interval, const double threshold, const double overlap,
			std::vector<Detection>& detections);
// Detect by loading model from filepath
void detect(const std::string mixture_filepath, const unsigned char* image,
            const int width, const int height, const int padding,
            const int interval, const double threshold, const double overlap,
			std::vector<Detection>& detections);
