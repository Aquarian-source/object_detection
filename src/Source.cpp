#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <cmath>
#include "Object_Detection.h"


using namespace std;
using namespace cv;
using namespace cv::dnn;


int main()
{

	Object_Detection Detect;
	
	//string cfg_path = "yolov3-tiny.cfg";
	//string weights_path = "yolov3-tiny.weights";
	string cfg_path = "yolov3.cfg";
	string weights_path = "yolov3.weights";



	//loading names of classes 
	string classesFile = "coco.names";
	ifstream ifs(classesFile.c_str());
	string line;
	while (getline(ifs, line)) Detect.classes.push_back(line);


	Net net = readNet(cfg_path, weights_path);


	while (waitKey(1) < 0) {
		Mat frame = imread("yo.png");

		//creating a blob to feed to neural network
		Mat blob = blobFromImage(frame, Detect.scale, cvSize(412, 412), Scalar(0, 0, 0), true, false);

		//Sets the input to the network
		net.setInput(blob);

		// Runs the forward pass to get output of the output layers
		vector<Mat> outs;
		net.forward(outs, Detect.getOutput(net));

		// Remove the bounding boxes with low confidence and draw bounding box with highest confidence
		Detect.postProcessing(frame, outs);
		

		namedWindow("Detection", WINDOW_NORMAL);
		imshow("Detection", frame);
	}
		

	return 0;
}
