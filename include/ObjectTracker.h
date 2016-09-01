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

private:
    std::vector<CompressiveTracker*> mvpTrackers;
    CompressiveTracker* mpBenchmarkTracker;
    CompressiveTracker* mpNewAlgoTracker;

    void processFrame2(cv::Mat& _frame, cv::Rect& _objectBox, int& radioMaxIndex, float& radioMax,ORB_SLAM2::Frame _currentFrame);

public:

    int mBenchmarkRadioMaxIndex;
    float mBenchmarkRadioMax;
    cv::Rect mBenchmarkObjectBox;

    int mNewAlgoTrackerRadioMaxIndex;
    float mNewAlgoTrackerRadioMax;
    cv::Rect mNewAlgoTrackerObjectBox;

    float mAreaPoints;
    float mArea;

    std::vector<int> mvRadioMaxIndexes;
    std::vector<float> mvRadioMaxes;
    std::vector<cv::Rect> mvObjectBoxes;

    int  mKeyTrackerRadioMaxIndex;
    float mKeyTrackerRadioMax;
    cv::Rect mKeyTrackerObjectBox;
    int mKeyTrackerIndex; 

    int mnTrackers;

};
