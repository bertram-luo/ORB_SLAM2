#include "ObjectTracker.h"
#include "Frame.h"
#define MAX_TRACKERS (30)
#include "MapPoint.h"
#include "slic.h"
#include "log.h"

ObjectTracker::ObjectTracker(ORB_SLAM2::FrameDrawer* frameDrawer, ORB_SLAM2::Tracking * tracker) :
    mpFrameDrawer(frameDrawer),
    mpTracker(tracker),
    mBenchmarkRadioMaxIndex(0),
    mBenchmarkRadioMax(0),
    mBenchmarkObjectBox(0, 0, 0, 0),
    mNewAlgoTrackerRadioMaxIndex(0),
    mNewAlgoTrackerRadioMax(0),
    mNewAlgoTrackerObjectBox(0, 0, 0, 0),
    mSPTrackerRadioMaxIndex(0),
    mSPTrackerRadioMax(0),
    mSPTrackerObjectBox(0, 0, 0, 0),
    mCtmsTrackerRadioMaxIndex(0),
    mCtmsTrackerRadioMax(0),
    mCtmsTrackerObjectBox(0, 0, 0, 0),
    mPointsArea(0),
    mArea(0),
    mvRadioMaxIndexes(std::vector<int>(2*(MAX_TRACKERS),0)),
    mvRadioMaxes(std::vector<float>(2*(MAX_TRACKERS), 0.f)),
    mvObjectBoxes(std::vector<cv::Rect>(2*(MAX_TRACKERS), cv::Rect(0, 0, 0, 0))),
    mnTrackers(-1)
{
        

    printf("creating object tracker\n");
    mpBenchmarkTracker = new CompressiveTracker;
    mpNewAlgoTracker = new CompressiveTracker;
    mvpTrackers.push_back(new CompressiveTracker);
    mpCtmsTracker = new CtmsTracker;
    mpSPTracker = new SPTracker;
    mnTrackers++;
    printf("object tracker created\n");

}


ObjectTracker::~ObjectTracker(){
    for_each(mvpTrackers.begin(), mvpTrackers.end(), [](CompressiveTracker* instance){
            delete instance;
            return nullptr;
            });

    delete mpBenchmarkTracker;
    delete mpNewAlgoTracker;
}

void ObjectTracker::init(cv::Mat& _oriFrame, cv::Mat& _frame, cv::Rect& _objectBox){
    mvObjectBoxes[0] = _objectBox;
    mBenchmarkObjectBox = _objectBox;
    mNewAlgoTrackerObjectBox = _objectBox;
    mCtmsTrackerObjectBox = _objectBox;
    mSPTrackerObjectBox = _objectBox;

    printf("===in init frame size[%d %d]\n", _frame.rows, _frame.cols);
    mvpTrackers[0]->init(_frame, mvObjectBoxes[0]);
    mpBenchmarkTracker->init(_frame, mBenchmarkObjectBox);
    mpNewAlgoTracker->init(_frame, mNewAlgoTrackerObjectBox);
    mpCtmsTracker->init(_frame, mCtmsTrackerObjectBox);
    printf("===in init frame size[%d %d]\n", _frame.rows, _frame.cols);
    mpSPTracker->init(_oriFrame, mSPTrackerObjectBox);
    printf("init done\n");
}

bool ObjectTracker::processFrameCtms(cv::Mat& _imOri, cv::Mat& _frame, ORB_SLAM2::Frame _currentFrame){
    _imOri.copyTo(mImOri);
    mpBenchmarkTracker->processFrame(_frame, mBenchmarkObjectBox , mBenchmarkRadioMaxIndex, mBenchmarkRadioMax);
    
    auto retval = newAlgoTrackingArea( _frame, _currentFrame);
    CtmsTracking(_frame, _currentFrame);
    return retval; 
}

bool ObjectTracker::CtmsTracking(cv::Mat _frame, ORB_SLAM2::Frame _currentFrame){
    cv::Mat prob(_frame.size(), CV_32F);
    prob.setTo(0);

    cv::Mat ori;
    _frame.convertTo(ori, CV_32FC1, 1/255.0);

    mpCtmsTracker->processFrame(_frame, mCtmsTrackerObjectBox, mCtmsTrackerRadioMaxIndex, mCtmsTrackerRadioMax, prob);

    printf("showing prob image\n");

    for(int r = 0; r < prob.rows; ++r){
        for(int c = 0; c < prob.cols; ++c){
            if (prob.at<float>(r, c) <= 0){
                prob.at<float>(r, c) = ori.at<float>(r, c);
            }
        }
    }
    cv::imshow("prob after",prob);
    cv::waitKey(2);
    return false; //TODO danger what is the bool retval for;
}

bool ObjectTracker::processFrameHeuristic(cv::Mat& _imOri, cv::Mat& _frame, ORB_SLAM2::Frame _currentFrame){
    _imOri.copyTo(mImOri);
    mpBenchmarkTracker->processFrame(_frame, mBenchmarkObjectBox , mBenchmarkRadioMaxIndex, mBenchmarkRadioMax);

    auto retval = newAlgoTrackingArea( _frame, _currentFrame);
    return retval;
}

bool ObjectTracker::processFrame(cv::Mat& _imOri, cv::Mat& _frame, ORB_SLAM2::Frame _currentFrame, ORB_SLAM2::Tracking* pTracker){
    //return processFrameHeuristic(_imOri, _frame, _currentFrame);
    printf("[%s: %d],int processFrame\n", __FILE__, __LINE__);
    cv::Mat new_frame = _frame.clone();
    processFrameCtms(_imOri, _frame, _currentFrame);
    processFrameSPT(_imOri, new_frame, _currentFrame, pTracker);
    SLAM_DEBUG("result by ctms [%d, %d, %d, %d]", mCtmsTrackerObjectBox.x, mCtmsTrackerObjectBox.y, mCtmsTrackerObjectBox.width, mCtmsTrackerObjectBox.height);
    SLAM_DEBUG("result by SPT [%d, %d, %d, %d]", mSPTrackerObjectBox.x, mSPTrackerObjectBox.y, mSPTrackerObjectBox.width, mSPTrackerObjectBox.height);
    return false;
}
bool ObjectTracker::processFrameSPT(cv::Mat& _imOri, cv::Mat& _frame, ORB_SLAM2::Frame _currentFrame, ORB_SLAM2::Tracking* pTracker){
    static int count = 0;
    static int skip_count = 0;
    skip_count ++;
    if (skip_count < 400){
        return false;
    }

    count ++;
    SLAM_DEBUG("in processFrameSPt %d\n", count);
    if (count <= 4){
        cv::Mat new_frame = _imOri.clone();
        mpSPTracker->addTrainFrame(new_frame, mCtmsTrackerObjectBox);
        if (count == 4){
            mpSPTracker->train();
        }
        return false;
    }
    mpSPTracker->run(_imOri, mSPTrackerObjectBox, pTracker);
}

//This colors the segmentations
void ObjectTracker::floodFillPostprocess( Mat& img, const Scalar& colorDiff)
{
    CV_Assert( !img.empty() );
    RNG rng = theRNG();
    Mat mask( img.rows+2, img.cols+2, CV_8UC1, Scalar::all(0) );
    for( int y = 0; y < img.rows; y++ )
    {
        for( int x = 0; x < img.cols; x++ )
        {
            if( mask.at<uchar>(y+1, x+1) == 0 )
            {
                Scalar newVal( rng(256), rng(256), rng(256) );
                floodFill( img, mask, Point(x,y), newVal, 0, colorDiff, colorDiff );
            }
        }
    }
}


bool ObjectTracker::newAlgoTrackingArea(cv::Mat& _frame, ORB_SLAM2::Frame _currentFrame){
    mpNewAlgoTracker->processFrameNotUpdateModel(_frame, mNewAlgoTrackerObjectBox , mNewAlgoTrackerRadioMaxIndex, mNewAlgoTrackerRadioMax);

    cv::Mat tmp1, tmp2, tmp3;
    mpFrameDrawer->DrawFrame(tmp1, tmp2, tmp3);
    cv::imshow("ORB-SLAM2: Current Frame",tmp1);
    waitKey(5);


    bool retval = false;
    bool done = false;
    int round = 0;

    bool scaled = false;
    for(;;){
        round ++;
        printf("-----------------round : %d--------------\n", round);
        calcPointAreaAndDirection(_currentFrame, 5);


        float x = mNewAlgoTrackerObjectBox.x;
        float y = mNewAlgoTrackerObjectBox.y;
        float w = mNewAlgoTrackerObjectBox.width;
        float h = mNewAlgoTrackerObjectBox.height;
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

        bool scaledLeft = scaled;
        bool scaledRight = scaled;
        bool scaledTop = scaled;
        bool scaledDown = scaled;

        printf("surround centroids [%f, %f, %f, %f], offset [%f, %f]\n", 
                m10XLess, m01YLess, m10XMore, m01YMore, offsetX, offsetY);
        printf( "left right up down: [%d %d %d %d], scaling facotr[%f, %f]\n", dXLess, dXMore, dYLess, dYMore, scaleX, scaleY);

        if (mPointsArea < mArea * 0.15){
            mpNewAlgoTracker->updateModel(_frame, mNewAlgoTrackerObjectBox, mNewAlgoTrackerRadioMaxIndex, mNewAlgoTrackerRadioMax);
            break;
        }

        cv::Rect tBB = mNewAlgoTrackerObjectBox;
        if (round >= 10){
            if (abs(dXLess - dXMore) < 5 && abs(dYLess - dYMore) < 5 && dXLess <= 9 && dXMore <= 9 && dYLess <= 9 && dYMore <= 9){
                break;
            }
            printf("ori location after round 10:[%d, %d, %d, %d]\n",mNewAlgoTrackerObjectBox.x, mNewAlgoTrackerObjectBox.y, mNewAlgoTrackerObjectBox.width, mNewAlgoTrackerObjectBox.height);
            if (abs(dXLess - dXMore) > 5 && (dXLess >=9 || dXMore >= 9)){
                if (dXLess > dXMore){
                    mNewAlgoTrackerObjectBox.x += dXLess - 9;
                    mNewAlgoTrackerObjectBox.width = mNewAlgoTrackerObjectBox.width - dXLess + 9;
                }else {
                    mNewAlgoTrackerObjectBox.width = mNewAlgoTrackerObjectBox.width - dXMore + 9;
                }
            }
            if (abs(dYLess - dYMore) > 5){
                if (dYLess > dYMore){
                    mNewAlgoTrackerObjectBox.y = mNewAlgoTrackerObjectBox.y + dYLess - 9;
                    mNewAlgoTrackerObjectBox.height = mNewAlgoTrackerObjectBox.height - dYLess + 9;
                }else {
                    mNewAlgoTrackerObjectBox.height = mNewAlgoTrackerObjectBox.height - dYMore + 9;
                }
            }
            printf("scaled location after round 10:[%d, %d, %d, %d]\n",mNewAlgoTrackerObjectBox.x, mNewAlgoTrackerObjectBox.y, mNewAlgoTrackerObjectBox.width, mNewAlgoTrackerObjectBox.height);
            mpNewAlgoTracker->updateModel(_frame, mNewAlgoTrackerObjectBox, mNewAlgoTrackerRadioMaxIndex, mNewAlgoTrackerRadioMax, true);
            retval = true;
            break;
        }

        if (round > 2){
            //if ((float)(dXMore)/mNewAlgoTrackerObjectBox.width > 0.15 && dXMore >= 15){
            if (dXMore >= 15){
                scaledLeft = true;
            }
            if (dXLess >= 15){
                scaledRight = true;
            }
            if (dYMore >= 15){
                scaledTop = true;
            }
            if(dYLess >= 15) {
                scaledDown = true;
            }
        }
        scaled = scaled || scaledTop || scaledDown || scaledLeft || scaledRight;


        if (scaledLeft){
            mNewAlgoTrackerObjectBox.width = mNewAlgoTrackerObjectBox.width - dXMore + 9;
        }
        if (scaledRight){
            mNewAlgoTrackerObjectBox.x = mNewAlgoTrackerObjectBox.x + dXLess - 9;
            mNewAlgoTrackerObjectBox.width = mNewAlgoTrackerObjectBox.width - dXLess + 9;
        }
        if (scaledTop){
            mNewAlgoTrackerObjectBox.height = mNewAlgoTrackerObjectBox.height - dYMore + 9;
        }
        if (scaledDown){
            mNewAlgoTrackerObjectBox.y = mNewAlgoTrackerObjectBox.y + dYLess - 9;
            mNewAlgoTrackerObjectBox.height = mNewAlgoTrackerObjectBox.height - dYLess + 9;
        }

        if (scaled){
            //waitKey(0);
            mpNewAlgoTracker->updateModel(_frame, mNewAlgoTrackerObjectBox, mNewAlgoTrackerRadioMaxIndex, mNewAlgoTrackerRadioMax, true);
            printf("scaled location [%d, %d, %d, %d]\n",mNewAlgoTrackerObjectBox.x, mNewAlgoTrackerObjectBox.y, mNewAlgoTrackerObjectBox.width, mNewAlgoTrackerObjectBox.height);
            //mpNewAlgoTracker->processFrameNotUpdateModel(_frame, mNewAlgoTrackerObjectBox , mNewAlgoTrackerRadioMaxIndex, mNewAlgoTrackerRadioMax);
            retval = true;
            break;
        }



        cv::Mat mOri;
        _frame.copyTo(mOri);
        rectangle(mOri, mNewAlgoTrackerObjectBox, cv::Scalar(0, 0, 255), 5);
        imshow("---ori--",mOri);
        waitKey(2);
        printf("==== ori [%d, %d, %d, %d]===\n", mNewAlgoTrackerObjectBox.x, mNewAlgoTrackerObjectBox.y,mNewAlgoTrackerObjectBox.width, mNewAlgoTrackerObjectBox.height);

        float centerX = (x + (float)w / 2) * scaleX;
        float centerY = (y + (float)h / 2) * scaleY;
        float originX = centerX - w / 2;
        float originY = centerY - h / 2;

        cv::Mat resized_frame;
        Size sz(cvRound((float)_frame.cols*scaleX), cvRound((float)_frame.rows*scaleY));
        resize(_frame, resized_frame, sz,0, 0,INTER_LINEAR);

        //mNewAlgoTrackerObjectBox.x = cvRound(m10XLess * scaleX);//use oriObjectBox;
        //mNewAlgoTrackerObjectBox.y = cvRound(m01YLess * scaleY);
        mNewAlgoTrackerObjectBox.x = originX;//use oriObjectBox;
        mNewAlgoTrackerObjectBox.y = originY;
        resized_frame.copyTo(mBefore);
        cv::Point pt1(mNewAlgoTrackerObjectBox.x,mNewAlgoTrackerObjectBox.y);
        cv::Point pt2(pt1.x + mNewAlgoTrackerObjectBox.width, pt1.y+mNewAlgoTrackerObjectBox.height);
        rectangle(mBefore, pt1,pt2,cv::Scalar(0,0,200));
        //printf("===before===[%d, %d, %d, %d]\n", pt1.x, pt1.y, pt2.x - pt1.x, pt2.y - pt1.y);
        printf("====before ===[%d, %d, %d, %d]\n", mNewAlgoTrackerObjectBox.x, mNewAlgoTrackerObjectBox.y,mNewAlgoTrackerObjectBox.width, mNewAlgoTrackerObjectBox.height);

        int leftRadius = dXMore * scaleX;
        int rightRadius = dXLess * scaleX;
        int upRadius = dYMore * scaleY;
        int downRadius = dYLess * scaleY;
        if (abs(dXLess - dXMore) < 5){
            leftRadius = 1;
            rightRadius = 1;
        }
        if (abs(dYMore - dYLess) < 5){
            upRadius = 1;
            downRadius = 1;
        }

        mpNewAlgoTracker->processFrameNotUpdateModel(
                resized_frame, mNewAlgoTrackerObjectBox ,
                mNewAlgoTrackerRadioMaxIndex, mNewAlgoTrackerRadioMax, 
                leftRadius, rightRadius, upRadius, downRadius);
        printf("=====after =====[%d, %d, %d, %d]\n", mNewAlgoTrackerObjectBox.x, mNewAlgoTrackerObjectBox.y,mNewAlgoTrackerObjectBox.width, mNewAlgoTrackerObjectBox.height);

        x = mNewAlgoTrackerObjectBox.x;
        y = mNewAlgoTrackerObjectBox.y;
        w = mNewAlgoTrackerObjectBox.width;
        h = mNewAlgoTrackerObjectBox.height;
        centerX = (x + (float)w / 2) / scaleX;
        centerY = (y + (float)h / 2) / scaleY;
        originX = centerX - w / 2;
        originY = centerY - h / 2;

        mNewAlgoTrackerObjectBox.x = cvRound(originX);
        mNewAlgoTrackerObjectBox.y = cvRound(originY);

        _frame.copyTo(mAfter);
        cv::Point pt3(mNewAlgoTrackerObjectBox.x,mNewAlgoTrackerObjectBox.y);
        cv::Point pt4(pt3.x + mNewAlgoTrackerObjectBox.width, pt3.y+mNewAlgoTrackerObjectBox.height);
        rectangle(mAfter, pt3,pt4,cv::Scalar(0,0, 200));
        printf("===== finally =====[%d, %d, %d, %d]\n", mNewAlgoTrackerObjectBox.x, mNewAlgoTrackerObjectBox.y,mNewAlgoTrackerObjectBox.width, mNewAlgoTrackerObjectBox.height);

        //mpNewAlgoTracker->init(_frame, mNewAlgoTrackerObjectBox);
        imshow("---before---", mBefore);
        imshow("---after---", mAfter);
        waitKey(1);
        //waitKey(0);

    }



    int spatialRad, colorRad, maxPyrLevel;
    spatialRad = 5;
    colorRad = 10;
    maxPyrLevel = 1;
    cv::Mat segres;
    //cv::Mat src(mImOri, mNewAlgoTrackerObjectBox);
    cv::Mat src(mImOri, mBenchmarkObjectBox);
    cv::Mat srcSLIC(mImOri, mBenchmarkObjectBox);
    
    pyrMeanShiftFiltering( src, segres, spatialRad, colorRad, maxPyrLevel );
    floodFillPostprocess( segres, Scalar::all(2) );
    src = segres;

    imshow("seged", src);
            waitKey(1);

    SLIC slic;
    slic.GenerateSuperpixels(srcSLIC, 175);

    cv::Mat slicResult = slic.GetImgWithContours(cv::Scalar(0, 0, 255));
    imshow("sliced", slicResult);
    //waitKey(0);


    return retval;
}

void ObjectTracker::checkSize(cv::Mat& frame, cv::Rect& BB, float& scaleX, float& scaleY){
    int xDiff = 0;
    int yDiff = 0;
    int w = BB.width;
    int h = BB.height;
    if (BB.x < 0){
       xDiff += BB.x;
       BB.x = 0;
    }

    if (BB.y > 0){
        yDiff += BB.y;
        BB.y = 0;
    
    }
    if (BB.x + BB.width > frame.cols){
        xDiff += BB.x + BB.width - frame.cols;
        BB.width = frame.cols - BB.x;
    
    }
    if (BB.y + BB.height > frame.rows){
        yDiff += BB.y + BB.height - frame.rows;
        BB.height = frame.cols - BB.y;
    }

    scaleX = (float)w / (w - xDiff);
    scaleY = (float)h / (h - yDiff);
}

void ObjectTracker::calcPointAreaAndDirection(ORB_SLAM2::Frame _currentFrame, int filter){

    float outerBoundFactor = 0.15;
    float innerBoundFactor = 0.25;

    float toleranceX = mNewAlgoTrackerObjectBox.width * 0.04;
    float toleranceY = mNewAlgoTrackerObjectBox.height * 0.04;
    int x1 = mNewAlgoTrackerObjectBox.x;
    int y1 = mNewAlgoTrackerObjectBox.y;
    int x2 = x1 + mNewAlgoTrackerObjectBox.width;
    int y2 = y1 + mNewAlgoTrackerObjectBox.height;
    float centroidX = (float)(x1 + x2) / 2;
    float centroidY = (float)(y1 + y2) / 2;
    float lowerBoundOuterX = ((float)x1 + outerBoundFactor * mNewAlgoTrackerObjectBox.width);
    float lowerBoundOuterY= ((float)y1  + outerBoundFactor * mNewAlgoTrackerObjectBox.height);
    float upperBoundOuterX = ((float)x1 + (1 - outerBoundFactor) * mNewAlgoTrackerObjectBox.width);
    float upperBoundOuterY = ((float)y1  + (1 - outerBoundFactor) * mNewAlgoTrackerObjectBox.height);
    float lowerBoundInnerX = ((float)x1 + innerBoundFactor * mNewAlgoTrackerObjectBox.width);
    float lowerBoundInnerY= ((float)y1  + innerBoundFactor * mNewAlgoTrackerObjectBox.height);
    float upperBoundInnerX = ((float)x1 + (1 - innerBoundFactor) * mNewAlgoTrackerObjectBox.width);
    float upperBoundInnerY = ((float)y1  + (1 - innerBoundFactor) * mNewAlgoTrackerObjectBox.height);
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
    float radius = sqrt(((float)(x2-x1)/2,2) + pow((float)(y2-y1)/2, 2));
    //float radius = sqrt((float)(x2-x1)/2 * (float)(y2-y1)/2);


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

                        if (pMP->Observations() >= 10){

                            if (x > lowerBoundOuterX && x < upperBoundOuterX && y < lowerBoundInnerY){
                                n01YLess++;
                                m01YLess += y - y1;
                            } 

                            if (x > lowerBoundOuterX && x < upperBoundOuterX && y > upperBoundInnerY ){
                                n01YMore++;
                                m01YMore += y - y1;
                            } 

                            if (y > lowerBoundOuterY && y < upperBoundOuterY && x < lowerBoundInnerX ){
                                n10XLess++;
                                m10XLess += x - x1;
                            } 

                            if (y > lowerBoundOuterY && y < upperBoundOuterY && x > upperBoundInnerX ){
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


