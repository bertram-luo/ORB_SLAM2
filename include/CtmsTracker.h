/************************************************************************
* File:	CtmsTracker
* Brief: Tracker of polo
* Version: 1.0
* Author: polo
* Email: luopo10164108@gmail.com
* Date:	2016/10/21
* History:
************************************************************************/
#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

using std::vector;
using namespace cv;
//---------------------------------------------------
class CtmsTracker
{
public:
	CtmsTracker(void);
	~CtmsTracker(void);

private:
	int featureMinNumRect;
	int featureMaxNumRect;
	int featureNum;
	vector<vector<Rect>> features;
	vector<vector<float>> featuresWeight;
	int rOuterPositive;
	vector<Rect> samplePositiveBox;
	vector<Rect> sampleNegativeBox;
	int rSearchWindow;
	Mat imageIntegral;
	Mat samplePositiveFeatureValue;
	Mat sampleNegativeFeatureValue;
	vector<float> muPositive;
	vector<float> sigmaPositive;
	vector<float> muNegative;
	vector<float> sigmaNegative;
	float learnRate;
	vector<Rect> detectBox;
	Mat detectFeatureValue;
	RNG rng;

private:
	void HaarFeature(Rect& _objectBox, int _numFeature);
	void sampleRect(Mat& _image, Rect& _objectBox, float _rInner, float _rOuter, int _maxSampleNum, vector<Rect>& _sampleBox);
	void sampleRect(Mat& _image, Rect& _objectBox, float _srw, vector<Rect>& _sampleBox);
	void getFeatureValue(Mat& _imageIntegral, vector<Rect>& _sampleBox, Mat& _sampleFeatureValue);
	void classifierUpdate(Mat& _sampleFeatureValue, vector<float>& _mu, vector<float>& _sigma, float _learnRate);
	void radioClassifier(vector<float>& _muPos, vector<float>& _sigmaPos, vector<float>& _muNeg, vector<float>& _sigmaNeg,
						Mat& _sampleFeatureValue, float& _radioMax, int& _radioMaxIndex, cv::Mat& _prob, vector<Rect>& detectBox);
public:
    void processFrame(Mat& _frame, Rect& _objectBox, int& radioMaxIndex, float& radioMax, cv::Mat& _prob);
	void init(Mat& _frame, Rect& _objectBox);
};
