#pragma once

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/highgui.hpp>
#include <fstream>

using namespace std;
using namespace cv;
using namespace cv::dnn;

class Object_Detection
{
public:

	Object_Detection();
	~Object_Detection();

	float conf_threshold = 0.5;
	float nms_threshold = 0.3;
	double scale = 0.002;
	vector<String> classes;


	void getPosition(float pos_x, float pos_y, float &resOut0, float& resOut1, float& dist, Mat);
	vector<String> getOutput(const cv::dnn::Net& net);
	void drawBoundingBox(int classId, float conf, int left, int top, int right, int bottom, Mat& frame, float dist);
	void postProcessing(Mat& frame, const vector<Mat>& outs);

};