#include "Object_Detection.h"



Object_Detection::Object_Detection()
{
}


void Object_Detection::getPosition(float pos_x, float pos_y, float &resOut0, float& resOut1, float& dist, Mat frame)
{
	// points taken from frame to which i want to do linear transformation
	Point2f pts1[4], pts2[4];
	pts1[0] = Point2f(237, 271);
	pts1[1] = Point2f(423, 271);
	pts1[2] = Point2f(638, 360);
	pts1[3] = Point2f(55, 360);
    
	
	pts2[0] = Point2f(-2.3, 10);
	pts2[1] = Point2f(2.3, 10);
	pts2[2] = Point2f(2.3, 2);
	pts2[3] = Point2f(-2.3, 2);
	
	Mat O;
	Mat M(3, 3, CV_32FC1);
	M = getPerspectiveTransform(pts1, pts2); 


	//Linear transformation of geometries
	Mat pos(3, 1, CV_32FC1);
	pos.at<float>(0, 0) = pos_x;
	pos.at<float>(1, 0) = pos_y;
	pos.at<float>(2, 0) = 1;

	Mat res(3, 1, CV_32FC1);

	for (int i = 0; i < M.rows; i++)
	{
		float val = 0;
		for (int j = 0; j < M.cols; j++)
		{
			val = (M.at<double>(i, j)) * (pos.at<float>(j, 0));
			val += val;
		}
		res.at<float>(i, 0) = val;
	}
	res.at<float>(0, 0) /= res.at<float>(2, 0);
	res.at<float>(1, 0) /= res.at<float>(2, 0);
	res.at<float>(2, 0) /= res.at<float>(2, 0);

	resOut0 = res.at<float>(0, 0);
	resOut1 = res.at<float>(1, 0);
	dist = sqrt(resOut0*resOut0 + resOut1 * resOut1);

}


vector<String> Object_Detection::getOutput(const Net& net)
{
	static vector<String> names;
	if (names.empty())
	{
		//Get the indices of the output layers, i.e. the layers with unconnected outputs
		vector<int> outLayers = net.getUnconnectedOutLayers();

		//get the names of all the layers in the network
		vector<String> layersNames = net.getLayerNames();

		// Get the names of the output layers in names
		names.resize(outLayers.size());
		for (size_t i = 0; i < outLayers.size(); ++i)
			names[i] = layersNames[outLayers[i] - 1];
	}
	return names;
}



void Object_Detection::drawBoundingBox(int classId, float conf, int left, int top, int right, int bottom, Mat& frame, float dist)
{
	//Draw a rectangle displaying the bounding box
	rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 0, 255));

	vector<String> classes;

	//Get the label for the class name and its confidence
	string label = format("%.2f", conf);
	if (!classes.empty())
	{
		CV_Assert(classId < (int)classes.size());
		label = classes[classId] + ":" + label;
	}
	//Display the label at the top of the bounding box
	int baseLine;
	Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
	top = max(top, labelSize.height);
	putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255));
	putText(frame, to_string(dist), Point(left, top-30), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255));

}


void Object_Detection::postProcessing(Mat& frame, const vector<Mat>& outs)
{
	vector<int> classIds;
	vector<float> confidences;
	vector<Rect> boxes;




	for (size_t i = 0; i < outs.size(); ++i)
	{
		// Scan through all the bounding boxes output from the network and keep only the
		// ones with high confidence scores. Assign the box's class label as the class
		// with the highest score for the box.
		float* data = (float*)outs[i].data;
		for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
		{
			Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
			Point classIdPoint;
			double confidence;

			// Get the value and location of the maximum score
			minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
			if (confidence > conf_threshold)
			{
				int centerX = (int)(data[0] * frame.cols);
				int centerY = (int)(data[1] * frame.rows);
				float width = (data[2] * frame.cols);
				float height = (data[3] * frame.rows);
				float left = centerX - width / 2;
				float top = centerY - height / 2;

				classIds.push_back(classIdPoint.x);
				confidences.push_back((float)confidence);
				boxes.push_back(Rect(left, top, width, height));

				float res0, res1, distance;

				getPosition(width + left / 2.0, height + top, res0, res1, distance, frame);
				cout << distance << endl;
				cout << "-----------------------" << endl;

				vector<int> indices;
				NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, indices);
				for (size_t i = 0; i < indices.size(); ++i)
				{
					int idx = indices[i];
					Rect box = boxes[idx];

					drawBoundingBox(classIds[idx], confidences[idx], box.x, box.y, box.x + box.width, box.y + box.height, frame, distance);
				}

			}

		}
	}
	
}

Object_Detection::~Object_Detection()
{
}
