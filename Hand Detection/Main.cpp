#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

#include <iostream>

using namespace cv;
using namespace std;

int hMin = 0;
int hMax = 256;
int sMin = 0;
int sMax = 256;
int vMin = 0;
int vMax = 256;
int sliderMax = 256;
void onTrackbar(int, void*) {}

const int FRAME_HEIGHT = 480;
const int FRAME_WIDTH = 640;
const int MIN_OBJECT_SIZE = 400;
const int MAX_OBJECT_SIZE = 76800;

void createTrackbars()
{
	namedWindow("HSV Properties", 0);
	createTrackbar("Hue Min", "HSV Properties", &hMin, sliderMax, onTrackbar);
	createTrackbar("Hue Max", "HSV Properties", &hMax, sliderMax, onTrackbar);
	createTrackbar("Saturation Min", "HSV Properties", &sMin, sliderMax, onTrackbar);
	createTrackbar("Saturation Max", "HSV Properties", &sMax, sliderMax, onTrackbar);
	createTrackbar("Value Min", "HSV Properties", &vMin, sliderMax, onTrackbar);
	createTrackbar("Value Max", "HSV Properties", &vMax, sliderMax, onTrackbar);
}

void morphOps(Mat &frame)
{

//	blur(frame, frame, Size(3, 3));
	GaussianBlur(frame, frame, Size(5, 5), 0, 0);

	erode(frame, frame, getStructuringElement(MORPH_RECT, Size(3,3)));	
	erode(frame, frame, getStructuringElement(MORPH_RECT, Size(3,3)));
//	erode(frame, frame, getStructuringElement(MORPH_RECT, Size(3,3)));

	dilate(frame, frame, getStructuringElement(MORPH_RECT, Size(8,8)));
	dilate(frame, frame, getStructuringElement(MORPH_RECT, Size(8,8)));
//	dilate(frame, frame, getStructuringElement(MORPH_RECT, Size(8,8)));
}

void cannyOps(Mat src, Mat dst)
{
	Mat edge;
	src.copyTo(edge);
	blur(edge, edge, Size(3, 3));
	Canny(src, edge, 100, 300, 3);
	edge.convertTo(dst, CV_8U);
	imshow("Canny Edge", dst);
}

void trackObject(Mat thresh, Mat raw, Mat cont, Mat HULL)
{
	Mat temp;
	thresh.copyTo(temp);

	vector< vector<Point> > contours;
	vector< Vec4i > hierarchy;

	findContours(temp, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

	vector<vector<Point> >hull(contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		convexHull(Mat(contours[i]), hull[i], false);
	}

	vector< Moments > mts(contours.size());
	for(int i = 0; i < contours.size(); i++)
	{
		mts[i] = moments(contours[i], false);
	}

	vector< Point2f > mc(contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		mc[i] = Point2f(mts[i].m10 / mts[i].m00, mts[i].m01 / mts[i].m00);
	}

	vector< Vec4i > cvxDefect;

	//int numObj = 0;
	double largestArea = 0;
	int largestContourIndex = 0;
	for(int i = 0; i < contours.size(); i++)
	{
		drawContours(raw, contours, i, Scalar(255, 255, 255), 1, 8, vector<Vec4i>(), 0, Point());
		drawContours(raw, hull, i, Scalar(255, 255, 255), 1, 8, vector<Vec4i>(), 0, Point());

		//convexityDefects(contours, hull, cvxDefect);
		

		double area;
		area = mts[i].m00;
		
		if (area > largestArea)
		{
			largestArea = area;
			largestContourIndex = i;
		}

		if (area > MIN_OBJECT_SIZE && area < MAX_OBJECT_SIZE)
		{
			circle(raw, mc[i], 10, Scalar(0, 0, 255), 1);
			if (mc[i].x > 20 && mc[i].x < FRAME_WIDTH - 20 && mc[i].y > 20 && mc[i].y < FRAME_HEIGHT - 20)
			{
				line(raw, Point(mc[i].x, mc[i].y + 15), Point(mc[i].x, mc[i].y - 15), Scalar(0, 0, 255), 2);
				line(raw, Point(mc[i].x - 15, mc[i].y), Point(mc[i].x + 15, mc[i].y), Scalar(0, 0, 255), 2);
			}
			//numObj++;
		}
	}
	
	Rect bounding_rect = boundingRect(contours[largestContourIndex]);
	rectangle(raw, bounding_rect, Scalar(255, 0, 0), 1, 8, 0);
	putText(raw, "Tracking...", Point(20, 50), 1, 2, Scalar(0, 0, 255), 2);
}


int main()
{
	Mat RAW;
	Mat HSV;
	Mat YCC;
	Mat threshold;
	Mat morph;
	Mat edges;
	Mat hull;

	VideoCapture webcam;
	webcam.open(0);
	webcam.set(CV_CAP_PROP_FRAME_WIDTH, FRAME_WIDTH);
	webcam.set(CV_CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT);

	createTrackbars();

	while (1)
	{
		webcam.read(RAW);
		cvtColor(RAW, HSV, COLOR_BGR2HSV);
//		cvtColor(RAW, HSV, COLOR_BGR2YCrCb);
//		blur(HSV, HSV, Size(3, 3));
		inRange(HSV, Scalar(hMin, sMin, vMin), Scalar(hMax, vMax, sMax), threshold);
		threshold.copyTo(morph);
		morphOps(morph);
		trackObject(morph, RAW, edges, hull);
		cannyOps(morph, edges);

//		imshow("YCC", YCC);
		imshow("Camera Feed", RAW);
//		imshow("HSV", HSV);
		imshow("Threshold", threshold);
//		imshow("Morphological Ops", morph);
//		imshow("Canny edges", edges);
//		imshow("Convex Hull", hull);
		
		waitKey(30);
	}

	return 0;
}
