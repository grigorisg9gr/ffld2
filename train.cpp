//--------------------------------------------------------------------------------------------------
// Implementation of the papers "Exact Acceleration of Linear Object Detectors", 12th European
// Conference on Computer Vision, 2012 and "Deformable Part Models with Individual Part Scaling",
// 24th British Machine Vision Conference, 2013.
//
// Copyright (c) 2013 Idiap Research Institute, <http://www.idiap.ch/>
// Written by Charles Dubout <charles.dubout@idiap.ch>
//
// This file is part of FFLDv2 (the Fast Fourier Linear Detector version 2)
//
// FFLDv2 is free software: you can redistribute it and/or modify it under the terms of the GNU
// Affero General Public License version 3 as published by the Free Software Foundation.
//
// FFLDv2 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
// the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero
// General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License along with FFLDv2. If
// not, see <http://www.gnu.org/licenses/>.
//--------------------------------------------------------------------------------------------------

#include "SimpleOpt.h"

#include "Mixture.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <cstdlib>

using namespace FFLD;
using namespace std;

ifstream in_pos_stream;    			// used for reading positive stream (non pascal)

int read_stream(ifstream& in, vector<Scene> (&scenes), int nbNegativeScenes, Object::Name name, string xml_end, int padding, int nonPascalPos, int& maxRows, int& maxCols, string folder);
int convert_name(string arg, Object::Name (&name));

// SimpleOpt array of valid options
enum
{
	OPT_C, OPT_DATAMINE, OPT_INTERVAL, OPT_HELP, OPT_J, OPT_RELABEL, OPT_MODEL, OPT_NAME,
	OPT_PADDING, OPT_RESULT, OPT_SEED, OPT_OVERLAP, OPT_NB_COMP, OPT_NB_NEG, OPT_PT_POS, 
	OPT_PT_POS_AN
};

CSimpleOpt::SOption SOptions[] =
{
	{ OPT_C, "-c", SO_REQ_SEP },
	{ OPT_C, "--C", SO_REQ_SEP },
	{ OPT_DATAMINE, "-d", SO_REQ_SEP },
	{ OPT_DATAMINE, "--datamine", SO_REQ_SEP },
	{ OPT_INTERVAL, "-e", SO_REQ_SEP },
	{ OPT_INTERVAL, "--interval", SO_REQ_SEP },
	{ OPT_HELP, "-h", SO_NONE },
	{ OPT_HELP, "--help", SO_NONE },
	{ OPT_J, "-j", SO_REQ_SEP },
	{ OPT_J, "--J", SO_REQ_SEP },
	{ OPT_RELABEL, "-l", SO_REQ_SEP },
	{ OPT_RELABEL, "--relabel", SO_REQ_SEP },
	{ OPT_MODEL, "-m", SO_REQ_SEP },
	{ OPT_MODEL, "--model", SO_REQ_SEP },
	{ OPT_NAME, "-n", SO_REQ_SEP },
	{ OPT_NAME, "--name", SO_REQ_SEP },
	{ OPT_PADDING, "-p", SO_REQ_SEP },
	{ OPT_PADDING, "--padding", SO_REQ_SEP },
	{ OPT_RESULT, "-r", SO_REQ_SEP },
	{ OPT_RESULT, "--result", SO_REQ_SEP },
	{ OPT_SEED, "-s", SO_REQ_SEP },
	{ OPT_SEED, "--seed", SO_REQ_SEP },
	{ OPT_OVERLAP, "-v", SO_REQ_SEP },
	{ OPT_OVERLAP, "--overlap", SO_REQ_SEP },
	{ OPT_NB_COMP, "-x", SO_REQ_SEP },
	{ OPT_NB_COMP, "--nb-components", SO_REQ_SEP },
	{ OPT_NB_NEG, "-z", SO_REQ_SEP },
	{ OPT_NB_NEG, "--nb-negatives", SO_REQ_SEP },
	{ OPT_PT_POS, "-dbp", SO_REQ_SEP },
	{ OPT_PT_POS, "--file-pos", SO_REQ_SEP },
	{ OPT_PT_POS_AN, "-dban", SO_REQ_SEP },
	{ OPT_PT_POS_AN, "--path-pos-an", SO_REQ_SEP },
	SO_END_OF_OPTIONS
};

void showUsage()
{
	cout << "Usage: train [options] image_set.txt\n\n"
			"Options:\n"
			"  -c,--C <arg>             SVM regularization constant (default 0.002)\n"
			"  -d,--datamine <arg>      Maximum number of data-mining iterations within each "
			"training iteration  (default 10)\n"
			"  -e,--interval <arg>      Number of levels per octave in the HOG pyramid (default 5)"
			"\n"
			"  -h,--help                Display this information\n"
			"  -j,--J <arg>             SVM positive regularization constant boost (default 2)\n"
			"  -l,--relabel <arg>       Maximum number of training iterations (default 8, half if "
			"no part)\n"
			"  -m,--model <file>        Read the initial model from <file> (default zero model)\n"
			"  -n,--name <arg>          Name of the object to detect (default \"person\")\n"
			"  -p,--padding <arg>       Amount of zero padding in HOG cells (default 6)\n"
			"  -r,--result <file>       Write the trained model to <file> (default \"model.txt\")\n"
			"  -s,--seed <arg>          Random seed (default time(NULL))\n"
			"  -v,--overlap <arg>       Minimum overlap in latent positive search (default 0.7)\n"
			"  -x,--nb-components <arg> Number of mixture components (without symmetry, default 3)\n"
			"  -z,--nb-negatives <arg>  Maximum number of negative images to consider (default all)\n"
			"  -dbp,--file-pos <file> 	The file that contains the positive samples (if different than pascal)\n"
			"  -dban,--path-pos-an <file> 	The folder with annotations for positive samples (if different than pascal)\n"
//			"  -dbp_name,--npp-n <arg> 	The name of the positive annotations (for different than pascal)\n"
		 << endl;
}

// Train a mixture model
int main(int argc, char * argv[])
{
	// Default parameters
	double C = 0.002;
	int nbDatamine = 10;
	int interval = 5;
	double J = 2.0;
	int nbRelabel = 8;
	string model;
	Object::Name name = Object::PERSON;
	int padding = 6;
	string result("model.txt");
	int seed = static_cast<int>(time(0));
	double overlap = 0.7;
	int nbComponents = 3;
	int nbNegativeScenes = -1;
	int nonPascalPos = 0;    			// if we want to load positives from a non-pascal dataset  
	string folder_non_pascal = "";      
	string path_non_pascal_anno ="";    // path for positive samples annotations (if different than pascal)
	
	// Parse the parameters
	CSimpleOpt args(argc, argv, SOptions);
	
	while (args.Next()) {
		if (args.LastError() == SO_SUCCESS) {
			if (args.OptionId() == OPT_C) {
				C = atof(args.OptionArg());
				
				if (C <= 0) {
					showUsage();
					cerr << "\nInvalid C arg " << args.OptionArg() << endl;
					return -1;
				}
			}
			else if (args.OptionId() == OPT_DATAMINE) {
				nbDatamine = atoi(args.OptionArg());
				
				if (nbDatamine <= 0) {
					showUsage();
					cerr << "\nInvalid datamine arg " << args.OptionArg() << endl;
					return -1;
				}
			}
			else if (args.OptionId() == OPT_INTERVAL) {
				interval = atoi(args.OptionArg());
				
				if (interval <= 0) {
					showUsage();
					cerr << "\nInvalid interval arg " << args.OptionArg() << endl;
					return -1;
				}
			}
			else if (args.OptionId() == OPT_HELP) {
				showUsage();
				return 0;
			}
			else if (args.OptionId() == OPT_J) {
				J = atof(args.OptionArg());
				
				if (J <= 0) {
					showUsage();
					cerr << "\nInvalid J arg " << args.OptionArg() << endl;
					return -1;
				}
			}
			else if (args.OptionId() == OPT_RELABEL) {
				nbRelabel = atoi(args.OptionArg());
				
				if (nbRelabel <= 0) {
					showUsage();
					cerr << "\nInvalid relabel arg " << args.OptionArg() << endl;
					return -1;
				}
			}
			else if (args.OptionId() == OPT_MODEL) {
				model = args.OptionArg();
			}
			else if (args.OptionId() == OPT_NAME) {
				string arg = args.OptionArg();
				transform(arg.begin(), arg.end(), arg.begin(), static_cast<int (*)(int)>(tolower));
				/*
				const string Names[21] =
				{
					"aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
					"cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
					"sheep", "sofa", "train", "tvmonitor", "my_class" 									// the last case for non-pascal positives
				};
				cout << arg << "   " << args.OptionArg() <<endl;
				const string * iter = find(Names, Names + 20, arg);										// in case that it's not categorised, it falls in the last "my_class", which is only for non-pascal
					
				if (iter == Names + 21) {
					showUsage();
					cerr << "\nInvalid name arg " << args.OptionArg() << endl;
					return -1;
				}
				
				name = static_cast<Object::Name>(iter - Names); */
				convert_name(arg,name);
			}
			else if (args.OptionId() == OPT_PADDING) {
				padding = atoi(args.OptionArg());
				
				if (padding <= 1) {
					showUsage();
					cerr << "\nInvalid padding arg " << args.OptionArg() << endl;
					return -1;
				}
			}
			else if (args.OptionId() == OPT_RESULT) {
				result = args.OptionArg();
			}
			else if (args.OptionId() == OPT_SEED) {
				seed = atoi(args.OptionArg());
			}
			else if (args.OptionId() == OPT_OVERLAP) {
				overlap = atof(args.OptionArg());
				
				if ((overlap <= 0.0) || (overlap >= 1.0)) {
					showUsage();
					cerr << "\nInvalid overlap arg " << args.OptionArg() << endl;
					return -1;
				}
			}
			else if (args.OptionId() == OPT_NB_COMP) {
				nbComponents = atoi(args.OptionArg());
				
				if (nbComponents <= 0) {
					showUsage();
					cerr << "\nInvalid nb-components arg " << args.OptionArg() << endl;
					return -1;
				}
			}
			else if (args.OptionId() == OPT_NB_NEG) {
				nbNegativeScenes = atoi(args.OptionArg());
				
				if (nbNegativeScenes < 0) {
					showUsage();
					cerr << "\nInvalid nb-negatives arg " << args.OptionArg() << endl;
					return -1;
				}
			}
			else if (args.OptionId() == OPT_PT_POS){
				folder_non_pascal = args.OptionArg();
				in_pos_stream.open(args.OptionArg());				// the path of the non pascal positive images' list 
				if (in_pos_stream.fail()){
					showUsage();
					cerr << "Sorry, the file of parsing different positives couldn't be opened!\n";
					return -1;
				}
				nonPascalPos = 1;
			}
			else if (args.OptionId() == OPT_PT_POS_AN){
				path_non_pascal_anno = args.OptionArg();
				// TODO: check if this is a valid path for annotations
			}
			
		}
		
		else {
			showUsage();
			cerr << "\nUnknown option " << args.OptionText() << endl;
			return -1;
		}
	}
	
	srand(seed);
	srand48(seed);
	
	if (!args.FileCount()) {
		showUsage();
		cerr << "\nNo dataset provided" << endl;
		return -1;
	}
	else if (args.FileCount() > 1) {
		showUsage();
		cerr << "\nMore than one dataset provided" << endl;
		return -1;
	}

	if (((name == static_cast<Object::Name>(20))||(name == static_cast<Object::Name>(21)))&&(!nonPascalPos)){			// confirm that it is a valid pascal class, if the positives are from PASCAL
		cerr << "\nCalled an unknown class for VOC Pascal" << endl;
		return -1;
	}
	
	// Open the image set file
	const string file(args.File(0));
	const size_t lastDot = file.find_last_of('.');
	
	if ((lastDot == string::npos) || (file.substr(lastDot) != ".txt")) {
		showUsage();
		cerr << "\nInvalid image set file " << file << ", should be .txt" << endl;
		return -1;
	}
	
	ifstream in(file.c_str());
	
	if (!in.is_open()) {
		showUsage();
		cerr << "\nInvalid image set file " << file << endl;
		return -1;
	}
	
	// Find the annotations' folder (not sure that will work under Windows)
	const string folder = file.substr(0, file.find_last_of("/\\")) + "/../../Annotations/";
	
	// Load all the scenes
	int maxRows = 0;
	int maxCols = 0;
	
	vector<Scene> scenes;

	read_stream(in,scenes,nbNegativeScenes, name, ".xml", padding, nonPascalPos, maxRows, maxCols, folder);
	if (nonPascalPos){
		convert_name("face",name);					// convert the name to show "face" for positives
		if (path_non_pascal_anno.length() < 5){			// then assume that the folder is in the same path as the non-pascal txt file, with the name of the clip
			path_non_pascal_anno = folder_non_pascal.substr(0, folder_non_pascal.find_last_of(".")) +"/"; 
		}
		read_stream(in_pos_stream,scenes,0, name, "_0.voc2007.xml", padding, 0, maxRows, maxCols, path_non_pascal_anno); //folder and name to be fixed
		cout << path_non_pascal_anno << "  "  <<folder_non_pascal << endl;
		
	}
	

	// Initialize the Patchwork class
	if (!Patchwork::InitFFTW((maxRows + 15) & ~15, (maxCols + 15) & ~15)) {
		cerr << "Error initializing the FFTW library" << endl;
		return - 1;
	}

	// The mixture to train
	Mixture mixture(nbComponents, scenes, name);
	if (mixture.empty()) {
		cerr << "Error initializing the mixture model" << endl;
		return -1;
	}
	
	// Try to open the mixture
	if (!model.empty()) {
		ifstream in(model.c_str(), ios::binary);
		
		if (!in.is_open()) {
			showUsage();
			cerr << "\nInvalid model file " << model << endl;
			return -1;
		}
		in >> mixture;
		if (mixture.empty()) {
			showUsage();
			cerr << "\nInvalid model file " << model << endl;
			return -1;
		}
	}
	
	if (model.empty())
		mixture.train(scenes, name, padding, padding, interval, nbRelabel / 2, nbDatamine, 24000, C,
					  J, overlap);
	if (mixture.models()[0].parts().size() == 1)
		mixture.initializeParts(8, make_pair(6, 6));
	
	mixture.train(scenes, name, padding, padding, interval, nbRelabel, nbDatamine, 24000, C, J,
				  overlap);
	
	// Try to open the result file
	ofstream out(result.c_str(), ios::binary);
	
	if (!out.is_open()) {
		showUsage();
		cerr << "\nInvalid result file " << result << endl;
		cout << mixture << endl; // Print the mixture as a last resort
		return -1;
	}
	
	out << mixture;
	
	return EXIT_SUCCESS;
}

int read_stream(ifstream& in, vector<Scene> (&scenes), int nbNegativeScenes, Object::Name name, string xml_end, int padding, int nonPascalPos, int& maxRows, int& maxCols, string folder){
	while (in) {
		string line;
		getline(in, line);
		
		// Skip empty lines
		if (line.size() < 3)
			continue;
		
		// Check whether the scene is positive or negative
		const Scene scene(folder + line.substr(0, line.find(' ')) + xml_end);
		
		if (scene.empty())
			continue;
		
		bool positive = false;
		bool negative = true;
		
		for (int i = 0; i < scene.objects().size(); ++i) {
			if (scene.objects()[i].name() == name) {
				negative = false;
				
				if ((!scene.objects()[i].difficult())&&(!nonPascalPos))
					positive = true;
			}
		}
		
		if (positive || (negative && nbNegativeScenes)) {
			scenes.push_back(scene); 
			maxRows = max(maxRows, (scene.height() + 3) / 4 + padding);
			maxCols = max(maxCols, (scene.width() + 3) / 4 + padding);
			
			if (negative)
				--nbNegativeScenes;
		}
	}
	
	if (scenes.empty()) {
		showUsage();
		cerr << "\nInvalid image_set file " << endl;
		exit(-1);
	}
	return 1;
}

 int convert_name(string arg, Object::Name (&name)){
    const string Names[22] =
    {
        "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
        "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
        "sheep", "sofa", "train", "tvmonitor", "face", "my_class" 									// the last 2 cases for non-pascal positives
    };
    const string * iter = find(Names, Names + 21, arg);										// in case that it's not categorised, it falls in the last "my_class", which is strictly for non-pascal positives

    if (iter == Names + 22) {
        showUsage();
        cerr << "\nInvalid name arg " << arg << endl;
        exit(-1);
    }

    name = static_cast<Object::Name>(iter - Names);
    return 1;
}
