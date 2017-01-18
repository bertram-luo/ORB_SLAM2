#ifndef OBJECTTRACKING_H
#define OBJECTTRACKING_H
#include <opencv2/core/core.hpp>
#include <vector>
#include "CompressiveTracker.h"
#include "CtmsTracker.h"
#include "Frame.h"
#include "FrameDrawer.h"
#include "SPTracker.h"
//#include "Tracking.h"

class ORB_SLAM2::Tracking;

class ObjectTracker{

public:
	ObjectTracker(ORB_SLAM2::FrameDrawer* frameDrawer, ORB_SLAM2::Tracking * tracker);
	~ObjectTracker(void);

public:
    bool processFrame(cv::Mat& _imOri, cv::Mat& _frame, ORB_SLAM2::Frame _currentFrame);
    void init(cv::Mat& _oriFrame, cv::Mat& _frame, cv::Rect& _objectBox);
    void newAlgoTrackingAreaAndNormalDirection(cv::Mat& _frame, ORB_SLAM2::Frame _currentFrame);


    void checkSize(cv::Mat& frame, cv::Rect& BB, float& scaleX, float& scaleY);

private:

    bool newAlgoTrackingArea(cv::Mat& _frame, ORB_SLAM2::Frame _currentFrame);

    bool CtmsTracking(cv::Mat _frame, ORB_SLAM2::Frame _currentFrame);
    bool processFrameSPT(cv::Mat& _imOri, cv::Mat& _frame, ORB_SLAM2::Frame _currentFrame);
    bool processFrameCtms(cv::Mat& _imOri, cv::Mat& _frame, ORB_SLAM2::Frame _currentFrame);
    bool processFrameHeuristic(cv::Mat& _imOri, cv::Mat& _frame, ORB_SLAM2::Frame _currentFrame);
    void floodFillPostprocess( Mat& img, const Scalar& colorDiff=Scalar::all(1) );
// depricated
    void processFrame2(cv::Mat& _frame, cv::Rect& _objectBox, int& radioMaxIndex, float& radioMax,ORB_SLAM2::Frame _currentFrame);

    void calcPointAreaAndDirection(ORB_SLAM2::Frame _currentFrame, int filter = 1);

    std::vector<CompressiveTracker*> mvpTrackers;
    CompressiveTracker* mpBenchmarkTracker;
    CompressiveTracker* mpNewAlgoTracker;
    SPTracker* mpSPTracker;
    CtmsTracker* mpCtmsTracker;

    ORB_SLAM2::FrameDrawer* mpFrameDrawer;
    ORB_SLAM2::Tracking* mpTracker;


public:

    int mBenchmarkRadioMaxIndex;
    float mBenchmarkRadioMax;
    cv::Rect mBenchmarkObjectBox;

    int mNewAlgoTrackerRadioMaxIndex;
    float mNewAlgoTrackerRadioMax;
    cv::Rect mNewAlgoTrackerObjectBox;


    int mCtmsTrackerRadioMaxIndex;
    float mCtmsTrackerRadioMax;
    cv::Rect mCtmsTrackerObjectBox;

    int mSPTrackerRadioMaxIndex;
    float mSPTrackerRadioMax;
    cv::Rect mSPTrackerObjectBox;

    cv::Mat mImOri;
    cv::Mat mBefore;
    cv::Mat mAfter;

    float mPointsArea;
    float mArea;
    float mPintsX;
    float mPointsY;

    int n01YLess;
    int n01YMore;
    int n10XLess;
    int n10XMore;

    float m01YLess;
    float m01YMore;
    float m10XLess;
    float m10XMore;


    std::vector<int> mvRadioMaxIndexes;
    std::vector<float> mvRadioMaxes;
    std::vector<cv::Rect> mvObjectBoxes;

    int  mKeyTrackerRadioMaxIndex;
    float mKeyTrackerRadioMax;
    cv::Rect mKeyTrackerObjectBox;
    int mKeyTrackerIndex; 

    int mnTrackers;

};
#endif
