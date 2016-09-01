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

void ObjectTracker::processFrame(cv::Mat& _frame, ORB_SLAM2::Frame _currentFrame){

    mpBenchmarkTracker->processFrame(_frame, mBenchmarkObjectBox , mBenchmarkRadioMaxIndex, mBenchmarkRadioMax);


    newAlgoTrackingArea(_frame, _currentFrame);
}



void ObjectTracker::newAlgoTrackingArea(cv::Mat& _frame, ORB_SLAM2::Frame _currentFrame){
    mpNewAlgoTracker->processFrameNotUpdateModel(_frame, mNewAlgoTrackerObjectBox , mNewAlgoTrackerRadioMaxIndex, mNewAlgoTrackerRadioMax);
    calcPointAreaAndDirection(_currentFrame, 5);

    if (mPointsArea > mArea * 0.15){
        cv::Mat resized_frame;


        printf("surround centroids [%f, %f], [%f, %f], [%f, %f], [%f, %f]\n", 
                m10XLessYLess, m01XLessYLess,
                m10XLessYMore, m01XLessYMore,
                m10XMoreYMore, m01XMoreYMore,
                m10XMoreYLess, m01XMoreYLess);
        float offsetX = (m10XLessYLess + m10XLessYMore)/2;
        float offsetY = (m10XLessYLess + m10XMoreYLess)/2;

        float scaleX = mNewAlgoTrackerObjectBox.width*1.0
            /((m10XMoreYLess + m10XMoreYMore)/2
             - (m10XLessYLess + m10XLessYMore)/2);
        float scaleY = mNewAlgoTrackerObjectBox.height * 1.0
            /((m01XLessYMore + m01XMoreYMore)/2
             - (m01XLessYLess + m01XMoreYLess)/2);

        //float scale = sqrt((float)mArea / (mArea - mPointsArea));
        //Size sz(cvRound((float)_frame.cols*scale), cvRound((float)_frame.rows*scale));
        
        
        printf("[%d, %d, %d, %d]\n", mNewAlgoTrackerObjectBox.x, mNewAlgoTrackerObjectBox.y,mNewAlgoTrackerObjectBox.width, mNewAlgoTrackerObjectBox.height);
        printf("scale factor:[%f, %f] \n", scaleX, scaleY);
        Size sz(cvRound((float)_frame.cols*scaleX), cvRound((float)_frame.rows*scaleY));
        resize(_frame, resized_frame, sz,0, 0,INTER_LINEAR);

        //mNewAlgoTrackerObjectBox.x = cvRound(mNewAlgoTrackerObjectBox.x * scale +mNewAlgoTrackerObjectBox.width * (scale -1) /2);//use oriObjectBox;
        //mNewAlgoTrackerObjectBox.y = cvRound(mNewAlgoTrackerObjectBox.y * scale + mNewAlgoTrackerObjectBox.width * (scale -1)/2);

        mNewAlgoTrackerObjectBox.x = cvRound(mNewAlgoTrackerObjectBox.x * scaleX + offsetX);//use oriObjectBox;
        mNewAlgoTrackerObjectBox.y = cvRound(mNewAlgoTrackerObjectBox.y * scaleY + offsetY);
        resized_frame.copyTo(mBefore);
        cv::Point pt1(mNewAlgoTrackerObjectBox.x,mNewAlgoTrackerObjectBox.y);
        cv::Point pt2(pt1.x + mNewAlgoTrackerObjectBox.width, pt1.y+mNewAlgoTrackerObjectBox.height);
        rectangle(mBefore, pt1,pt2,cv::Scalar(0,0,255));
        printf("[%d, %d, %d, %d]\n", pt1.x, pt1.y, pt2.x - pt1.x, pt2.y - pt1.y);

        //mpNewAlgoTracker->processFrameNotUpdateModel(_frame, mNewAlgoTrackerObjectBox , mNewAlgoTrackerRadioMaxIndex, mNewAlgoTrackerRadioMax);
        //mNewAlgoTrackerObjectBox.x = cvRound((float)mNewAlgoTrackerObjectBox.x / scale);
        //mNewAlgoTrackerObjectBox.y = cvRound((float)mNewAlgoTrackerObjectBox.y / scale);
        //mNewAlgoTrackerObjectBox.width = cvRound((float)mNewAlgoTrackerObjectBox.width / scale);
        //mNewAlgoTrackerObjectBox.height = cvRound((float)mNewAlgoTrackerObjectBox.height / scale);


        mNewAlgoTrackerObjectBox.x = cvRound((float)mNewAlgoTrackerObjectBox.x / scaleX);
        mNewAlgoTrackerObjectBox.y = cvRound((float)mNewAlgoTrackerObjectBox.y / scaleY);
        mNewAlgoTrackerObjectBox.width = cvRound((float)mNewAlgoTrackerObjectBox.width / scaleX);
        mNewAlgoTrackerObjectBox.height = cvRound((float)mNewAlgoTrackerObjectBox.height / scaleY);
        printf("[%d, %d, %d, %d]\n", mNewAlgoTrackerObjectBox.x, mNewAlgoTrackerObjectBox.y,mNewAlgoTrackerObjectBox.width, mNewAlgoTrackerObjectBox.height);


        _frame.copyTo(mAfter);
        cv::Point pt3(mNewAlgoTrackerObjectBox.x,mNewAlgoTrackerObjectBox.y);
        cv::Point pt4(pt3.x + mNewAlgoTrackerObjectBox.width, pt3.y+mNewAlgoTrackerObjectBox.height);
        rectangle(mAfter, pt3,pt4,cv::Scalar(0,0, 255));

        mpNewAlgoTracker->init(_frame, mNewAlgoTrackerObjectBox);
    } else {
        mpNewAlgoTracker->updateModel(_frame, mNewAlgoTrackerObjectBox , mNewAlgoTrackerRadioMaxIndex, mNewAlgoTrackerRadioMax);
    }

}

void ObjectTracker::calcPointAreaAndDirection(ORB_SLAM2::Frame _currentFrame, int filter){

    int x1 = mNewAlgoTrackerObjectBox.x;
    int y1 = mNewAlgoTrackerObjectBox.y;
    int x2 = x1 + mNewAlgoTrackerObjectBox.width;
    int y2 = y1 + mNewAlgoTrackerObjectBox.height;
    float centroidX = (float)(x1 + x2) / 2;
    float centroidY = (float)(y1 + y2) / 2;
    float firstQuantileX = ((float)x1 * 3.0 / 4 + 1.0/4 * x2);
    float firstQuantileY = ((float)y1 * 3.0 / 4 + 1.0/4 * y2);
    float thirdQuantileX = ((float)x1 * 1.0 / 4 + 3.0/4 * x2);
    float thirdQuantileY = ((float)y1 * 1.0 / 4 + 3.0/4 * y2);

    float lessPointsArea = 0;
    mPointsArea =0;
    npoints = 0;

    m10XLessYLess = 0;
    m01XLessYLess = 0;
    nXLessYLess = 0;

    m10XLessYMore = 0;
    m01XLessYMore = 0;
    nXLessYMore = 0;

    m10XMoreYLess = 0;
    m01XMoreYLess = 0;
    nXMoreYLess = 0;

    m10XMoreYMore = 0;
    m01XMoreYMore = 0;
    nXMoreYMore = 0;
    
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
                    if ( x > firstQuantileX && x < thirdQuantileX && y > firstQuantileY && y < thirdQuantileY){

                        nDiscardInnerPoints ++;
                        continue;
                    
                    }
                    if ( x >= x1 && x <= x2 && y >= y1 && y<=y2){
                        float dist = sqrt(pow((x - (float)(x1+x2)/2),2) + pow((y - (float)(y1+y2)/2), 2));
                        float s = 1 - dist/radius;


                        if (pMP->Observations() >= filter){
                            npoints++;
                            mPointsArea += 400 * s;

                            // calc centroid of points;
                            //

                            if (x < firstQuantileX && y < firstQuantileY ){
                                nXLessYLess ++;
                                m10XLessYLess += x -x1;
                                m01XLessYLess += y -y1;
                            } 
                            if (x < firstQuantileX && y > thirdQuantileY ){
                                nXLessYMore ++;
                                m10XLessYMore += x -x1;
                                m01XLessYMore += y -y1;
                            } 
                            if (x > thirdQuantileX && y < firstQuantileY ){
                                nXMoreYLess ++;
                                m10XMoreYLess += x -x1;
                                m01XMoreYLess += y -y1;
                            } 
                            if (x > thirdQuantileX && y > firstQuantileY ){
                                nXMoreYMore ++;
                                m10XMoreYMore += x -x1;
                                m01XMoreYMore += y -y1;
                            } 
                        } else {
                            lessPointsArea += 400 * s;
                        }
                        printf("%d, ", pMP->Observations());
                    }
                }
            }
        }
    }



    m10XLessYLess = m10XLessYLess / max(nXLessYLess,1) + x1;
    m01XLessYLess = m01XLessYLess / max(nXLessYLess,1) + y1;

    m10XLessYMore = m10XLessYMore / max(nXLessYMore,1) + x1;
    m01XLessYMore /= max(nXLessYMore,1);
    if (m01XLessYMore == 0) m01XLessYMore = y2;
    else m01XLessYMore += y1;

    m10XMoreYLess /= max(nXMoreYLess,1);
    if(m10XMoreYLess == 0) m10XMoreYLess = x2;
    else m10XMoreYLess += x1;
    m01XMoreYLess =  m01XMoreYLess / max(nXMoreYLess,1) + y1;

    m10XMoreYMore /= max(nXMoreYMore,1);
    if (m10XMoreYMore ==0) m10XMoreYMore = x2;
    else m10XMoreYMore += x1;
    m01XMoreYMore /= max(nXMoreYMore,1);
    if (m01XMoreYMore ==0) m01XMoreYMore = y2;
    else m01XMoreYMore += y1;

    printf("\n");
    printf("LPA%f,PA %f, ratio %f |TA %f \n",
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


