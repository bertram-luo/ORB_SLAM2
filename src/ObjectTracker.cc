#include "ObjectTracker.h"

ObjectTracker::ObjectTracker() : mBenchmarkRadioMaxIndex(0), mBenchmarkRadioMax(0), mvRadioMaxIndexes(std::vector<int>(1,0)),mvRadioMaxes(std::vector<float>(1, 0.f)), mvObjectBoxes(std::vector<cv::Rect>(1, cv::Rect(0, 0, 0, 0))) {

    mpBenchmarkTracker = new CompressiveTracker;
    mvpTrackers.push_back(new CompressiveTracker);

}


ObjectTracker::~ObjectTracker(){
    for_each(mvpTrackers.begin(), mvpTrackers.end(), [](CompressiveTracker* instance){
            delete instance;
            return nullptr;
            });

    delete mpBenchmarkTracker;
}

void ObjectTracker::init(cv::Mat& _frame, cv::Rect& _objectBox){
    mvpTrackers[0]->init(_frame, _objectBox);
    mpBenchmarkTracker->init(_frame, _objectBox);
}

void ObjectTracker::processFrame(cv::Mat& _frame, cv::Rect& _objectBox, int& radioMaxIndex, float& radioMax){

    cv::Rect oriObjectBox = _objectBox;
    mpBenchmarkTracker->processFrame(_frame, _objectBox, radioMaxIndex, radioMax);
    mBenchmarkObjectBox = _objectBox;
    mBenchmarkRadioMaxIndex = radioMaxIndex;
    mBenchmarkRadioMax = radioMax;


    if (mBenchmarkRadioMax > 0){
        for(int i = 0; i < mvpTrackers.size(); ++i){
            mvObjectBoxes[i] = oriObjectBox;
            mvpTrackers[i]->processFrameNotUpdateModel(_frame, mvObjectBoxes[i], mvRadioMaxIndexes[i], mvRadioMaxes[i]);
        }
    } else {

        for(int i = 0; i < mvpTrackers.size(); ++i){
            mvObjectBoxes[i] = oriObjectBox;
            mvpTrackers[i]->processFrameNotUpdateModel(_frame, mvObjectBoxes[i], mvRadioMaxIndexes[i], mvRadioMaxes[i]);
            mvpTrackers[i]->fullImageScan(_frame, mvObjectBoxes[i], mvRadioMaxIndexes[i], mvRadioMaxes[i]);
        }

        int i = distance(mvRadioMaxes.begin(), max_element(mvRadioMaxes.begin(), mvRadioMaxes.end()));
        _objectBox = mvObjectBoxes[i];

        mpBenchmarkTracker->init(_frame, _objectBox);
    }
}

