#include "CompressiveTracker.h"
#include <opencv2/core/core.hpp>
#include <vector>

class ObjectTracker{

public:
	ObjectTracker(void);
	~ObjectTracker(void);

public:
    void processFrame(cv::Mat& _frame, cv::Rect& _objectBox, int& radioMaxIndex, float& radioMax);
	void init(cv::Mat& _frame, cv::Rect& _objectBox);

private:
    std::vector<CompressiveTracker*> mvpTrackers;
    CompressiveTracker* mpBenchmarkTracker;


public:

    int mBenchmarkRadioMaxIndex;
    float mBenchmarkRadioMax;
    cv::Rect mBenchmarkObjectBox;


    std::vector<int> mvRadioMaxIndexes;
    std::vector<float> mvRadioMaxes;
    std::vector<cv::Rect> mvObjectBoxes;

    int  mKeyTrackerRadioMaxIndex;
    float mKeyTrackerRadioMax;
    cv::Rect mKeyTrackerObjectBox;
    int mKeyTrackerIndex; 

    int mnTrackers;

};
