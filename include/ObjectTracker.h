#include "CompressiveTracker.h"
#include <opencv2/core/core.hpp>
#include <vector>
#include "Frame.h"

class ObjectTracker{

public:
	ObjectTracker(void);
	~ObjectTracker(void);

public:
    void processFrame(cv::Mat& _frame, ORB_SLAM2::Frame _currentFrame);
	void init(cv::Mat& _frame, cv::Rect& _objectBox);
    void newAlgoTrackingArea(cv::Mat& _frame, ORB_SLAM2::Frame _currentFrame);
    void newAlgoTrackingAreaAndNormalDirection(cv::Mat& _frame, ORB_SLAM2::Frame _currentFrame);

private:
    std::vector<CompressiveTracker*> mvpTrackers;
    CompressiveTracker* mpBenchmarkTracker;
    CompressiveTracker* mpNewAlgoTracker;

    void processFrame2(cv::Mat& _frame, cv::Rect& _objectBox, int& radioMaxIndex, float& radioMax,ORB_SLAM2::Frame _currentFrame);

    void calcPointAreaAndDirection(ORB_SLAM2::Frame _currentFrame, int filter = 1);

public:

    int mBenchmarkRadioMaxIndex;
    float mBenchmarkRadioMax;
    cv::Rect mBenchmarkObjectBox;

    int mNewAlgoTrackerRadioMaxIndex;
    float mNewAlgoTrackerRadioMax;
    cv::Rect mNewAlgoTrackerObjectBox;

    cv::Mat mBefore;
    cv::Mat mAfter;

    float mPointsArea;
    float mArea;
    float mPintsX;
    float mPointsY;


    int npoints;
    float m10XLessYLess;
    float m01XLessYLess;
    int nXLessYLess;


    float m10XLessYMore;
    float m01XLessYMore;
    int nXLessYMore;


    float m10XMoreYLess;
    float m01XMoreYLess;
    int nXMoreYLess;


    float m10XMoreYMore;
    float m01XMoreYMore;
    int nXMoreYMore;


    std::vector<int> mvRadioMaxIndexes;
    std::vector<float> mvRadioMaxes;
    std::vector<cv::Rect> mvObjectBoxes;

    int  mKeyTrackerRadioMaxIndex;
    float mKeyTrackerRadioMax;
    cv::Rect mKeyTrackerObjectBox;
    int mKeyTrackerIndex; 

    int mnTrackers;

};
