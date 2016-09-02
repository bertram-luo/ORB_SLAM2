#include "ObjectTracker.h"
#include "Frame.h"
#define MAX_TRACKERS (30)
#include "MapPoint.h"

ObjectTracker::ObjectTracker() :
    mBenchmarkRadioMaxIndex(0),
    mBenchmarkRadioMax(0),
    mBenchmarkObjectBox(0, 0, 0, 0),
    mNewAlgoTrackerRadioMaxIndex(0),
    mNewAlgoTrackerRadioMax(0),
    mNewAlgoTrackerObjectBox(0, 0, 0, 0),
    mPointsArea(0),
    mArea(0),
    mvRadioMaxIndexes(std::vector<int>(2*(MAX_TRACKERS),0)),
    mvRadioMaxes(std::vector<float>(2*(MAX_TRACKERS), 0.f)),
    mvObjectBoxes(std::vector<cv::Rect>(2*(MAX_TRACKERS), cv::Rect(0, 0, 0, 0))),
    mnTrackers(-1)
{
        

    mpBenchmarkTracker = new CompressiveTracker;
    mpNewAlgoTracker = new CompressiveTracker;
    mvpTrackers.push_back(new CompressiveTracker);
    mnTrackers++;

}


ObjectTracker::~ObjectTracker(){
    for_each(mvpTrackers.begin(), mvpTrackers.end(), [](CompressiveTracker* instance){
            delete instance;
            return nullptr;
            });

    delete mpBenchmarkTracker;
    delete mpNewAlgoTracker;
}

void ObjectTracker::init(cv::Mat& _frame, cv::Rect& _objectBox){
    mvObjectBoxes[0] = _objectBox;
    mBenchmarkObjectBox = _objectBox;
    mNewAlgoTrackerObjectBox = _objectBox;

    mvpTrackers[0]->init(_frame, mvObjectBoxes[0]);
    mpBenchmarkTracker->init(_frame, mBenchmarkObjectBox);
    mpNewAlgoTracker->init(_frame, mNewAlgoTrackerObjectBox);
    printf("init done\n");
}

bool ObjectTracker::processFrame(cv::Mat& _frame, ORB_SLAM2::Frame _currentFrame){

    mpBenchmarkTracker->processFrame(_frame, mBenchmarkObjectBox , mBenchmarkRadioMaxIndex, mBenchmarkRadioMax);


    return newAlgoTrackingArea(_frame, _currentFrame);
}



bool ObjectTracker::newAlgoTrackingArea(cv::Mat& _frame, ORB_SLAM2::Frame _currentFrame){
    mpNewAlgoTracker->processFrameNotUpdateModel(_frame, mNewAlgoTrackerObjectBox , mNewAlgoTrackerRadioMaxIndex, mNewAlgoTrackerRadioMax);

    bool done = false;
    int round = 0;
    calcPointAreaAndDirection(_currentFrame, 3);
    while(!done){
        round ++;
        if (mPointsArea > mArea * 0.1){
            cv::Mat resized_frame;

            float offsetX = m10XLess - mNewAlgoTrackerObjectBox.x; 
            float offsetY = m01YLess - mNewAlgoTrackerObjectBox.y; 

            float scaleX = mNewAlgoTrackerObjectBox.width*1.0
                /( m10XMore - m10XLess);
            float scaleY = mNewAlgoTrackerObjectBox.height * 1.0
                /(m01YMore - m01YLess);

            int dXLess = m10XLess - mNewAlgoTrackerObjectBox.x;
            int dXMore = mNewAlgoTrackerObjectBox.width + mNewAlgoTrackerObjectBox.x - m10XMore;
            int dYLess = m01YLess - mNewAlgoTrackerObjectBox.y;
            int dYMore = mNewAlgoTrackerObjectBox.height + mNewAlgoTrackerObjectBox.y - m01YMore;
            printf("surround centroids [%f, %f, %f, %f], offset [%f, %f]\n", 
                    m10XLess, m01YLess, m10XMore, m01YMore, offsetX, offsetY);
            printf( "left right up down: [%d %d %d %d]\n", dXLess, dXMore, dYLess, dYMore);
            int radius = max(max(dXLess, dXMore), max(dYLess, dYMore));
            
            printf("scale factor:[%f, %f], search radius %d \n", scaleX, scaleY, radius);
            printf("====== ori =====[%d, %d, %d, %d]\n", mNewAlgoTrackerObjectBox.x, mNewAlgoTrackerObjectBox.y,mNewAlgoTrackerObjectBox.width, mNewAlgoTrackerObjectBox.height);

            Size sz(cvRound((float)_frame.cols*scaleX), cvRound((float)_frame.rows*scaleY));
            resize(_frame, resized_frame, sz,0, 0,INTER_LINEAR);

            mNewAlgoTrackerObjectBox.x = cvRound(mNewAlgoTrackerObjectBox.x * scaleX + offsetX + radius);//use oriObjectBox;
            mNewAlgoTrackerObjectBox.y = cvRound(mNewAlgoTrackerObjectBox.y * scaleY + offsetY + radius);
            resized_frame.copyTo(mBefore);
            cv::Point pt1(mNewAlgoTrackerObjectBox.x,mNewAlgoTrackerObjectBox.y);
            cv::Point pt2(pt1.x + mNewAlgoTrackerObjectBox.width, pt1.y+mNewAlgoTrackerObjectBox.height);
            rectangle(mBefore, pt1,pt2,cv::Scalar(0,0,200));
            //printf("===before===[%d, %d, %d, %d]\n", pt1.x, pt1.y, pt2.x - pt1.x, pt2.y - pt1.y);
            printf("====before ===[%d, %d, %d, %d]\n", mNewAlgoTrackerObjectBox.x, mNewAlgoTrackerObjectBox.y,mNewAlgoTrackerObjectBox.width, mNewAlgoTrackerObjectBox.height);


            mpNewAlgoTracker->processFrameNotUpdateModel(_frame, mNewAlgoTrackerObjectBox , mNewAlgoTrackerRadioMaxIndex, mNewAlgoTrackerRadioMax, radius);
            printf("=====after =====[%d, %d, %d, %d]\n", mNewAlgoTrackerObjectBox.x, mNewAlgoTrackerObjectBox.y,mNewAlgoTrackerObjectBox.width, mNewAlgoTrackerObjectBox.height);

            
            mNewAlgoTrackerObjectBox.x = cvRound((float)mNewAlgoTrackerObjectBox.x / scaleX);
            mNewAlgoTrackerObjectBox.y = cvRound((float)mNewAlgoTrackerObjectBox.y / scaleY);
            if (dXLess == 0 || dXMore == 0 || round == 1){ scaleX = 1;}
            if (dYLess == 0 || dYMore == 0 || round == 1) {scaleY = 1;}
            mNewAlgoTrackerObjectBox.width = cvRound((float)mNewAlgoTrackerObjectBox.width / scaleX);
            mNewAlgoTrackerObjectBox.height = cvRound((float)mNewAlgoTrackerObjectBox.height / scaleY);

            _frame.copyTo(mAfter);
            cv::Point pt3(mNewAlgoTrackerObjectBox.x,mNewAlgoTrackerObjectBox.y);
            cv::Point pt4(pt3.x + mNewAlgoTrackerObjectBox.width, pt3.y+mNewAlgoTrackerObjectBox.height);
            rectangle(mAfter, pt3,pt4,cv::Scalar(0,0, 200));

            printf("===== finally =====[%d, %d, %d, %d]\n", mNewAlgoTrackerObjectBox.x, mNewAlgoTrackerObjectBox.y,mNewAlgoTrackerObjectBox.width, mNewAlgoTrackerObjectBox.height);

            
            calcPointAreaAndDirection(_currentFrame, 3);
            if (mPointsArea < mArea * 0.1){
                mpNewAlgoTracker->updateModel(_frame, mNewAlgoTrackerObjectBox, mNewAlgoTrackerRadioMaxIndex, mNewAlgoTrackerRadioMax);
                return true;
            }
            //mpNewAlgoTracker->init(_frame, mNewAlgoTrackerObjectBox);
        } else {
            mpNewAlgoTracker->updateModel(_frame, mNewAlgoTrackerObjectBox, mNewAlgoTrackerRadioMaxIndex, mNewAlgoTrackerRadioMax);
            return false;
        }
    }

    return true;
}

void ObjectTracker::calcPointAreaAndDirection(ORB_SLAM2::Frame _currentFrame, int filter){


    float toleranceX = mNewAlgoTrackerObjectBox.width * 0.04;
    float toleranceY = mNewAlgoTrackerObjectBox.height * 0.04;
    int x1 = mNewAlgoTrackerObjectBox.x;
    int y1 = mNewAlgoTrackerObjectBox.y;
    int x2 = x1 + mNewAlgoTrackerObjectBox.width;
    int y2 = y1 + mNewAlgoTrackerObjectBox.height;
    float centroidX = (float)(x1 + x2) / 2;
    float centroidY = (float)(y1 + y2) / 2;
    float lowerBoundX = ((float)x1 + 0.1 * mNewAlgoTrackerObjectBox.width);
    float lowerBoundY = ((float)y1  + 0.1 * mNewAlgoTrackerObjectBox.height);
    float upperBoundX = ((float)x1 + 0.9 * mNewAlgoTrackerObjectBox.width);
    float upperBoundY = ((float)y1  + 0.9 * mNewAlgoTrackerObjectBox.height);
    float firstQuantileX = (3.0 / 4 * x1 + 1.0 / 4 * x2);
    float firstQuantileY = (3.0 / 4 * y1 + 1.0 / 4 * y2);
    float thirdQuantileX = (1.0 / 4 * x1 + 3.0 / 4 * x2);
    float thirdQuantileY = (1.0 / 4 * y1 + 3.0 / 4 * y2);

    float lessPointsArea = 0;
    mPointsArea =0;
    int npoints = 0;

    n01YLess = 0;
    n01YMore = 0;
    n10XLess = 0;
    n10XMore = 0;

    m01YLess = 0;
    m01YMore = 0;
    m10XLess = 0;
    m10XMore = 0;

    
    mArea = mNewAlgoTrackerObjectBox.width * mNewAlgoTrackerObjectBox.height;
    float radius = sqrt(pow((float)(x2-x1)/2,2) + pow((float)(y2-y1)/2, 2));


    int nDiscardInnerPoints  = 0;
    for(int i = 0; i < (int)_currentFrame.mvpMapPoints.size(); ++i){
        int x = _currentFrame.mvKeysUn[i].pt.x;
        int y = _currentFrame.mvKeysUn[i].pt.y;
        ORB_SLAM2::MapPoint* pMP = _currentFrame.mvpMapPoints[i];
        if(pMP)
        {
            if(!_currentFrame.mvbOutlier[i])
            {
                if (pMP->Observations() >= 1){

                    if ( x >= x1 + toleranceX && x <= x2 - toleranceX && y >= y1 + toleranceY && y<=y2 - toleranceY){

                        if (pMP->Observations() >= filter){

                            if (x > lowerBoundX && x < upperBoundX && y < lowerBoundY ){
                                n01YLess++;
                                m01YLess += y - y1;
                            } 

                            if (x > lowerBoundX && x < upperBoundX && y > upperBoundY ){
                                n01YMore++;
                                m01YMore += y - y1;
                            } 

                            if (y > lowerBoundY && y < upperBoundY && x < lowerBoundX ){
                                n10XLess++;
                                m10XLess += x - x1;
                            } 

                            if (y > lowerBoundY && y < upperBoundY && x > upperBoundX ){
                                n10XMore++;
                                m10XMore += x - x1;
                            } 

                        }
                    }
                }
            }
        }
    }


    n01YLess = n01YLess > 2 ? n01YLess : INT_MAX;
    n01YMore = n01YMore > 2 ? n01YMore : INT_MAX;
    n10XLess = n10XLess > 2 ? n10XLess : INT_MAX;
    n10XMore = n10XMore > 2 ? n10XMore : INT_MAX;

    m01YLess = m01YLess / n01YLess + y1;
    m01YMore /= n01YMore;
    if (m01YMore < 1e-6) m01YMore = y2;
    else m01YMore += y1;
    m10XLess = m10XLess / n10XLess + x1;
    m10XMore /= n10XMore;
    if (m10XMore < 1e-6) m10XMore = x2;
    else m10XMore += x1;

    for(int i = 0; i < (int)_currentFrame.mvpMapPoints.size(); ++i){
        int x = _currentFrame.mvKeysUn[i].pt.x;
        int y = _currentFrame.mvKeysUn[i].pt.y;
        ORB_SLAM2::MapPoint* pMP = _currentFrame.mvpMapPoints[i];
        if(pMP)
        {
            if(!_currentFrame.mvbOutlier[i])
            {
                if (pMP->Observations() >= 1){

                    if ( x >= x1 + toleranceX && x <= x2 - toleranceX && y >= y1 + toleranceY && y<=y2 - toleranceY){

                        if ( x > m10XLess + m10XLess - x1 && x < m10XMore - x2 + m10XMore  && y > m01YLess + m01YLess - y1 && y < m01YMore - y2 + m01YMore){
                            continue;
                        }

                        float dist = sqrt(pow((x - (float)(x1+x2)/2),2) + pow((y - (float)(y1+y2)/2), 2));
                        float s = 1 - dist/radius;

                        if (pMP->Observations() >= filter){
                            npoints++;
                            mPointsArea += 400 * s;

                        } else {
                            lessPointsArea += 400 * s;
                        }
                    }
                }
            }
        }
    }






    printf("\n");
    printf("points:%d LPA%f,PA %f, ratio %f |TA %f \n",
            npoints,
            lessPointsArea, 
            mPointsArea,
            lessPointsArea/mPointsArea,
            mArea);
}

void ObjectTracker::newAlgoTrackingAreaAndNormalDirection(cv::Mat& _frame, ORB_SLAM2::Frame _currentFrame){

}


void ObjectTracker::processFrame2(cv::Mat& _frame, cv::Rect& _objectBox, int& radioMaxIndex, float& radioMax, ORB_SLAM2::Frame _currentFrame){

    cv::Rect oriObjectBox = _objectBox;
    mpBenchmarkTracker->processFrameNotUpdateModel(_frame, _objectBox, radioMaxIndex, radioMax);
    mBenchmarkObjectBox = _objectBox;
    mBenchmarkRadioMaxIndex = radioMaxIndex;
    mBenchmarkRadioMax = radioMax;


    int x1 = _objectBox.x;
    int y1 = _objectBox.y;
    int x2 = x1 + _objectBox.width;
    int y2 = y1 + _objectBox.height;
    int area_points =0;
    int area = _objectBox.width * _objectBox.height;
    float radius = sqrt(pow((float)(x2-x1)/2,2) + pow((float)(y2-y1)/2, 2));
    for(int i = 0; i < (int)_currentFrame.mvpMapPoints.size(); ++i){
        int x = _currentFrame.mvKeysUn[i].pt.x;
        int y = _currentFrame.mvKeysUn[i].pt.y;
        ORB_SLAM2::MapPoint* pMP = _currentFrame.mvpMapPoints[i];
        if(pMP)
        {
            if(!_currentFrame.mvbOutlier[i])
            {
                if(pMP->Observations()>0){
                    if ( x >= x1 && x <= x2 && y >= y1 && y<=y2){
                        float dist = sqrt(pow((x - (float)(x1+x2)/2),2) + pow((y - (float)(y1+y2)/2), 2));
                        float s = 1 - dist/radius;
                        area_points += 400 * s;
                    }
                }
            }
        }
    }

    printf("area_points %d, area %d\n", area_points, area);
    if (area_points > area * 0.1){
        cv::Mat resized_frame;

        float scale = sqrt((float)area / (area - area_points));
        Size sz(cvRound((float)_frame.cols*scale), cvRound((float)_frame.rows*scale));
        resize(_frame, resized_frame, sz,0, 0,INTER_LINEAR);
        _objectBox.x = cvRound((float)_objectBox.x * scale);//use oriObjectBox;
        _objectBox.y = cvRound((float)_objectBox.x * scale);
        mpBenchmarkTracker->processFrame(resized_frame, _objectBox, radioMaxIndex, radioMax);
        _objectBox.x = cvRound((float)_objectBox.x / scale);
        _objectBox.y = cvRound((float)_objectBox.y / scale);
        _objectBox.width = cvRound((float)_objectBox.width / scale);
        _objectBox.height = cvRound((float)_objectBox.height / scale);
        mBenchmarkObjectBox = _objectBox;
        mBenchmarkRadioMaxIndex = radioMaxIndex;
        mBenchmarkRadioMax = radioMax;
    }


    if (mBenchmarkRadioMax > 0){

        mKeyTrackerRadioMax = -INT_MAX;
        for(int i = 0; i <= mnTrackers; ++i){
            mvObjectBoxes[i] = oriObjectBox;
            mvpTrackers[i]->processFrameNotUpdateModel(_frame, mvObjectBoxes[i], mvRadioMaxIndexes[i], mvRadioMaxes[i]);
            if (mvRadioMaxes[i] > mKeyTrackerRadioMax){
                mKeyTrackerRadioMax = mvRadioMaxes[i];
                mKeyTrackerRadioMaxIndex = mvRadioMaxIndexes[i];
                mKeyTrackerObjectBox = mvObjectBoxes[i];
                mKeyTrackerIndex = i;
            }
        }

        if (mKeyTrackerRadioMax < -1500){// IDEA: fixed value is bad, could I make it auto adjusted? CT may do bad in log ratio
            if (mnTrackers >= MAX_TRACKERS){
                //TODO bug of libc mem management module?
                delete *(mvpTrackers.begin() + (int)(MAX_TRACKERS/2));
                for(int i = (int)(MAX_TRACKERS/2); i < mnTrackers;++i){
                    mvObjectBoxes[i] = mvObjectBoxes[i+1];
                    mvRadioMaxIndexes[i] = mvRadioMaxIndexes[i+1];
                    mvRadioMaxes[i] = mvRadioMaxIndexes[i+1];
                    mvpTrackers[i] = mvpTrackers[i+1];
                }
                mnTrackers--;
            }

            mnTrackers++;
            mvpTrackers[mnTrackers] = (new CompressiveTracker);
            mvObjectBoxes[mnTrackers] = (mBenchmarkObjectBox);
            mvRadioMaxIndexes[mnTrackers] = 0;
            mvRadioMaxes[mnTrackers] = 0;
            mvpTrackers[mnTrackers]->init(_frame, mvObjectBoxes[mnTrackers]);
            mvpTrackers[mnTrackers]->processFrameNotUpdateModel(_frame, mvObjectBoxes[mnTrackers], mvRadioMaxIndexes[mnTrackers], mvRadioMaxes[mnTrackers]);


            mKeyTrackerRadioMax = mvRadioMaxes[mnTrackers];
            mKeyTrackerRadioMaxIndex = mvRadioMaxIndexes[mnTrackers];
            mKeyTrackerObjectBox = mvObjectBoxes[mnTrackers];
            mKeyTrackerIndex = mnTrackers;
        
        }
    } else {

        for(int i = 0; i <= mnTrackers; ++i){
            mvObjectBoxes[i] = oriObjectBox;
            mvpTrackers[i]->processFrameNotUpdateModel(_frame, mvObjectBoxes[i], mvRadioMaxIndexes[i], mvRadioMaxes[i]);
            mvpTrackers[i]->fullImageScan(_frame, mvObjectBoxes[i], mvRadioMaxIndexes[i], mvRadioMaxes[i]);
        }

        int i = distance(mvRadioMaxes.begin(), max_element(mvRadioMaxes.begin(), mvRadioMaxes.begin() + mnTrackers + 1));
        _objectBox = mvObjectBoxes[i];

        mpBenchmarkTracker->init(_frame, _objectBox);
    }
}


