#include "SPTracker.h"
#include "slic.h"
#include <math.h>
#include <random>
#include <iterator>
#include "log.h"
#include <numeric>
#include <algorithm>
#include "MapPoint.h"
#include <set>
#include <chrono>
//TODO check o-indexing 1-indexing for matlab index from 1; c++  index from 0

int statSuperpixel(int* labels, int w, int h, vector<int>& sp_no);
void rgb2hsi(cv::Mat& image, cv::Mat& res);
void find_min_sp_index_and_max_distance(vector<int>& temp_nearest_train_sp_index, vector<float>& temp_fatherest_train_sp_distance, vector<vector<float>>& cluster_dis);
void segByProb(cv::Mat& pro_image_w1_255, cv::Mat& seg);
void setProbs(cv::Mat& pro_image_w1, int *labels, int rows, int cols, vector<float>& w1);
void find_max_spt_conf_location(cv::Mat& pro_image_w1, int w, int h, int& max_row, int& max_col, double& max_spt_conf);

void check_sum(std::string title, std::vector<std::vector<float>>& sp_cl_hist);

void find_vals_and_sps(cv::Mat& im, int x, int y, float _fill_value, SPTracker::SPTFrame& frame, std::set<float>& vals, std::set<int>& sps);
void floodFill(cv::Mat& im, int x, int y, std::set<float>& val);
void filterOutValue(cv::Mat& pro_image_w1, int rows, int cols, float threshold);
void fill_value(cv::Mat& im, float fill_value, std::set<float>& vals, std::set<int>& sps);

void find_weight(vector<float>& w2, cv::Mat& pro_image_w1, SPTracker::SPTFrame& test);
void sp_gauss_blur(vector<float>& w2, std::vector<std::set<int>>& adjacent_table, int level);
int SPTracker::SPTFrame::SPTFrame_instance_no = 0;
SPTracker::~SPTracker(){

}
SPTracker::SPTracker():
    numsample(600)
{


    /* beginning of trackparam.m*/
    grid_size = 64;
    grid_ratio = 1.1;
    train_frame_num = 0;
    show_patch_num = 5;


    HSI_flag = 1;
    ch_bins_num = 8;
    cluster_bandWidth = 0.18;
    SLIC_spatial_proximity_weight = 10;
    SLIC_sp_num = 500;
    negative_penality_ratio = 3.0;
    sigma_c = 7.6;
    sigma_s = 7;

    train_sp_sum_num = 0;
    train_sum_hist.resize(pow(ch_bins_num, 3));
    train_sum_hist_index;
    frame_num = 0;
    offline_incre_num = 1;
    update_flag = 0;
    update_sp_sum_num = 0;
    train_index_pre.push_back(0);

    update_incre_num = 15;
    update_frequency= 10;
    update_spacing = 1;
    occlusion_rate = 0.515;

    // ?? sp_pos_cluster, sp_neg_cluster, sp_mid_cluster;

    //p = [115, 252, 68, 72,0]
    numsample = 600;
    affsig = {8, 8, 0, 0, 0, 0};
    tmplsize = std::make_pair(grid_size, grid_size);
    // param0 = ; param0 = affparam2mat(param0);
    /* end of trackparam.m*/

    param0.resize(6, 0.0);
    numDim = 512;
    // ??
    update_num = 0;
}

void SPTracker::init(cv::Mat& frame, cv::Rect bb){
    addTrainFrame(frame, bb);
    cv::Rect _bb(bb);
    dump_rect(_bb.x, _bb.y, _bb.width, _bb.height);
    _bb.x = bb.x + bb.width / 2;
    _bb.y = bb.y + bb.height / 2;
    p[0] = _bb.x;
    p[1] = _bb.y;
    param0[0] = p[0];
    param0[1] = p[1];
    param0[2] = p[2]/tmplsize.second;
    param0[3] = p[5];
    param0[4] = p[4]/p[3];
    img_width = frame.cols;// frame.w
    img_height = frame.rows;// frame.h
    int lambda_r = (img_width + img_height) / 2;// 
    mot_sig_v1 = sigma_c * lambda_r;//TODO ??
    mot_sig_v2 = sigma_s * lambda_r;//TODO ?? find the corresponding part in paper
    //maxbasis, batchsize, errorfunc, ff, minopt
    //
    //temp_hsi_image = rgb2hsi(iframe);
    //templ.mean_Hue
    //tmpl.mean_Sat
    //tmpl.mean_Inten
    //templ.mean = warpimg(temp_frame, param0, opt.tmplsize);
    //tmpl.basis
    //tmpl.eigval
    //tmpl.numsample
    //tmpl.reseig
    //size = size(tmpl.mean);
    //N = sz(1)* sz(2);
    //param.est = param0;
    //param.wimg = tmpl.mean;
    //
    //train_box_param = zeros(6, myopt.train_frame_num);
    //train_box_param(:, 1) = param0;
    //drawopt.showcondens = 0;
    //grawopt.thcondens = 1 / opt.numsample;
}
void SPTracker::addTrainFrame(cv::Mat frame, cv::Rect newone){
    SLAM_DEBUG("entering add train frame");
    dump_rect(newone.x, newone.y, newone.width, newone.height);
    newone.x = newone.x + newone.width/2;
    newone.y = newone.y + newone.height/2;
    groundtruth_boxes.push_back(newone);
    train_frame_num ++;
    train_frames.resize(train_frame_num);
    train_frames[train_frame_num - 1].image = frame.clone();
    train_frames[train_frame_num - 1].p[0] = newone.x;
    train_frames[train_frame_num - 1].p[1] = newone.y;
    train_frames[train_frame_num - 1].p[2] = newone.width;
    train_frames[train_frame_num - 1].p[3] = newone.height;

}
void SPTracker::train(){
    SLAM_DEBUG("========start training========");
    for(unsigned int i = 0; i < groundtruth_boxes.size(); ++i){
        SLAM_DEBUG("--------processing frame %d, frame size [%d, %d]", i, img_height, img_width);

        train_sp_sum_num += train_frames[i].prepare_info(grid_ratio, SLIC_sp_num);
        train_index_pre.push_back(train_sp_sum_num);

        vector<pair<int, int>> temp_index;
        t1_cal_hsi_hist(train_frames[i], i, temp_index);
        copy(temp_index.begin(), temp_index.end(), back_inserter(train_sum_hist_index));
        addHist(train_sum_hist, train_frames[i].sp_cl_hist);
    }
    map_background.clear();
    map_background.resize(train_index_pre.back(), 0);
    SLAM_DEBUG("========before constrcut appearance model");
    t1_construct_appearance_model(train_sum_hist, train_sp_sum_num);//TODO check matlab code what should test be
    //after training, setting update parameters;
    int update_num = train_frame_num;
    update_frames.resize(update_num);
    for(int frameNo = 0; frameNo < update_num; ++ frameNo){
        //update_frames[frameNo].labels = (int*)malloc(sizeof(int) * train_frames[frameNo].warpimg_hsi.rows * train_frames[frameNo].warpimg_hsi.cols);
        //memcpy(update_frames[frameNo].labels, train_frames[frameNo].labels, train_frames[frameNo].warpimg_hsi.rows * train_frames[frameNo].warpimg_hsi.cols * sizeof(int));
        update_frames[frameNo].labels = train_frames[frameNo].labels;
        train_frames[frameNo].labels = nullptr;
        update_frames[frameNo].warp_p= train_frames[frameNo].warp_p;
        //update[frameNo].warpimg_tmpl = train_frames[frameNo].warpimg_tmpl;
        update_frames[frameNo].warpimg_hsi = train_frames[frameNo].warpimg_hsi;
    }
    update_hist_sum = train_sum_hist;
    update_sp_num = train_index_pre[update_num];
    update_index_pre = train_index_pre;
    update_index_pre_final = train_index_pre;//!!IMPORTANT different from original code
    update_index_pre.erase(update_index_pre.begin());

    frame_num = groundtruth_boxes.size();

    update_interval_num = train_frame_num;
    SLAM_DEBUG("========train completed");
}

void SPTracker::run(cv::Mat& frame, cv::Rect& new_location, ORB_SLAM2::Tracking* pTracker){
    //last_box_param = train_box_param(myopt.train_frame_num, :);
    copyMapPoints(pTracker);
    int update_negative = 0;
    frame_num ++;//TODO

    //auto start_tracking= chrono::steady_clock::now();
    do_tracking(frame, pTracker);
    //auto end_tracking= chrono::steady_clock::now();
    //auto diff = end_tracking- start_tracking;
    //std::cout << "tracking consumes"<< chrono::duration <double, milli> (diff).count() << " ms" << std::endl;

    new_location = test.to_tl();
    //new_location.x = test.p[0] - test.p[2] / 2;
    //new_location.y = test.p[1] - test.p[3] / 2;
    //new_location.width = test.p[2];
    //new_location.height = test.p[3];

    t1_update_info();
    SLAM_DEBUG("interval num %d frequ %d %d %d", update_interval_num, update_frequency, update_index_pre.size(), update_incre_num);
    if (update_interval_num >= update_frequency && update_index_pre.size() == update_incre_num){
        SLAM_DEBUG("=====updating model===");
        //auto start_update = chrono::steady_clock::now();
        t1_update_app_model();
        //auto end_update = chrono::steady_clock::now();
        //auto diff = end_update - start_update;
        //std::cout << "update model consumes"<< chrono::duration <double, milli> (diff).count() << " ms" << std::endl;

    }
    SLAM_DEBUG("spt tracker run completed");
}

int statSuperpixel(int* labels, int w, int h, vector<int>& sp_no){
    int* _labels = (int*)malloc(sizeof(int) * w * h);
    memcpy(_labels, labels, w*h*sizeof(int));
    sort(_labels, _labels + w * h);
    sp_no.push_back(*_labels);
    for(int index = 1; index < w * h; ++index){
        if (*(_labels + index) != sp_no.back()){
            sp_no.push_back(*(_labels+index)); 
        }
    }
    delete _labels;
    return sp_no.size();
}


void SPTracker::calc_sp_w1(vector<int> &temp_nearest_train_sp_index, vector<float>& temp_fatherest_train_sp_distance, SPTFrame& frame, vector<float>& w1){
    int samecent = 0;
    for(int spNo = 0; spNo < frame.sp_num; ++spNo){
        int cent = data2cluster[temp_nearest_train_sp_index[spNo]];
        samecent = cent;
        //printf("%d, ", cent);
        float dist = exp(-2*sqrt(pointDistanceSdColumn(frame.sp_cl_hist, spNo, clustCent, cent)) / temp_fatherest_train_sp_distance[spNo]);
        w1[spNo] = dist * train_cluster_weight[cent].first ;
    }
    //printf("\n");
    //for(int dim = 0; dim < 512; ++dim){
    //printf("%f", clustCent[dim][samecent]);
    //}
    //printf("\n");
    SLAM_DEBUG("sp weight computation completed");
    //printf("\n");
    //std::copy(w1.begin(), w1.end(), std::ostream_iterator<float>(std::cout, ", "));
    //printf("\n");
}

void SPTracker::do_tracking(cv::Mat& frame, ORB_SLAM2::Tracking* pTracker){
    SLAM_DEBUG("previous location [%f, %f, %f, %f, %f, %f]", test.p[0], test.p[1], test.p[2], test.p[3], test.p[4], test.p[5]);
    //-----prepare info
    //auto start_prepare_info = chrono::steady_clock::now();
    test.image=frame.clone();
    test.prepare_info(grid_ratio, SLIC_sp_num);
    //auto end_prepare_info = chrono::steady_clock::now();
    //auto diff = end_prepare_info- start_prepare_info;
    //std::cout << "prepare_info consumes"<< chrono::duration <double, milli> (diff).count() << " ms" << std::endl;
    imshow("sliced roi", test.imgWithContours);
    waitKey(1);

    for(int i = 0; i < mvCurrentKeys.size(); ++i){
        mvCurrentKeys[i].pt.x -= test.x1;
        mvCurrentKeys[i].pt.y -= test.y1;
    }

    //-----pixel scoring
    auto start_pixel_scoring = chrono::steady_clock::now();
    vector<pair<int, int>> temp_index;
    t1_cal_hsi_hist(test, frame_num, temp_index);//TODO

    auto start_pointsetdistance = chrono::steady_clock::now();
    vector<vector<float>> cluster_dis;
    pointSetDistance(test.sp_cl_hist, train_sum_hist, cluster_dis);
    auto end_pointsetdistance = chrono::steady_clock::now();
    auto diff_pointsetdistance = end_pointsetdistance - start_pointsetdistance;
    std::cout << "pointsetdistance consumes"<< chrono::duration <double, milli> (diff_pointsetdistance).count() << " ms" << std::endl;


    auto start_nearest_distance = chrono::steady_clock::now();

    vector<int> temp_nearest_train_sp_index;
    vector<float> temp_fatherest_train_sp_distance;
    find_min_sp_index_and_max_distance(temp_nearest_train_sp_index, temp_fatherest_train_sp_distance, cluster_dis);
    auto end_nearest_distance = chrono::steady_clock::now();
    auto diff_nearest_distance = end_nearest_distance- start_nearest_distance;
    std::cout << "nearest_distance consumes"<< chrono::duration <double, milli> (diff_nearest_distance).count() << " ms" << std::endl;


    auto start_calc_w1 = chrono::steady_clock::now();
    vector<float> w1(test.sp_num, 1);
    calc_sp_w1(temp_nearest_train_sp_index, temp_fatherest_train_sp_distance, test, w1);
    auto end_calc_w1 = chrono::steady_clock::now();
    auto diff_calc_w1 = end_calc_w1- start_calc_w1;
    std::cout << "calc_w1 consumes"<< chrono::duration <double, milli> (diff_calc_w1).count() << " ms" << std::endl;


    auto start_setprob = chrono::steady_clock::now();

    cv::Mat pro_image_w1(test.warpimg_hsi.rows, test.warpimg_hsi.cols, CV_32F);
    setProbs(pro_image_w1, test.labels, test.warpimg_hsi.rows, test.warpimg_hsi.cols, w1);
    auto end_setprob = chrono::steady_clock::now();
    auto diff_setprob = end_setprob- start_setprob;
    std::cout << "setprob consumes"<< chrono::duration <double, milli> (diff_setprob).count() << " ms" << std::endl;
    SLAM_DEBUG("pixel scoring completed");
    imshow("prob", pro_image_w1);
    waitKey(1);

    double min_prob;
    double max_prob;
    cv::minMaxIdx(pro_image_w1, &min_prob, &max_prob);
    SLAM_DEBUG("min max prob [%f, %f]", float(min_prob), float(max_prob));
    cv::Mat pro_image_w1_ori = pro_image_w1.clone();
    cv::Mat pro_image_w1_ori_255;
    pro_image_w1_ori.convertTo(pro_image_w1_ori_255, CV_8UC1, 255, 0);
    auto end_pixel_scoring = chrono::steady_clock::now();
    auto diff_pixel_scoring = end_pixel_scoring- start_pixel_scoring;
    std::cout << "pixel_scoring consumes"<< chrono::duration <double, milli> (diff_pixel_scoring).count() << " ms" << std::endl;

    utilizeMapPoints(pro_image_w1, test.x1, test.y1, min_prob, test);

    imshow("prob after mappoint", pro_image_w1);
    waitKey(1);


    pro_image_w1 += (-min_prob);
    pro_image_w1 /= (max_prob - min_prob);
    filterOutValue(pro_image_w1, test.warpimg_hsi.rows, test.warpimg_hsi.cols, -min_prob/(max_prob - min_prob));

    vector<float> w2(test.sp_num, 1);
    find_weight(w2, pro_image_w1, test);
    sp_gauss_blur(w2, test.adjacent_table, 1);
    setProbs(pro_image_w1, test.labels, test.warpimg_hsi.rows, test.warpimg_hsi.cols, w2);

    imshow("prob after gauss blur", pro_image_w1);
    waitKey(1);

    cv::Mat pro_image_w1_255;
    pro_image_w1.convertTo(pro_image_w1_255, CV_8UC1, 255, 0);
    cv::Mat seg = test.warpimg.clone();
    segByProb(pro_image_w1_255, seg);
    imshow("prob seged", seg);
    waitKey(1);

    cv::Mat hotmap;
    applyColorMap(pro_image_w1, hotmap, COLORMAP_JET);
    imshow("hotmap", hotmap);
    waitKey(1);

    int w = test.p[2];
    int h = test.p[3];
    int max_row = 0;
    int max_col = 0;
    double max_spt_conf = 0;
    find_max_spt_conf_location(pro_image_w1, w, h, max_row, max_col, max_spt_conf);
    SLAM_DEBUG("[%d] frame, sample regions spt confidence calculation completed with rect [%d, %d, %f, %f]", frame_num, int(max_col + w / 2), int(max_row + h / 2), w, h);
    dump_rect(max_col + test.x1, max_row + test.y1, w, h);
    test.p[0] = max_col + w / 2 + test.x1;
    test.p[1] = max_row + h / 2 + test.y1;
    SLAM_DEBUG(" [%d] frame, new object location [%f, %f, %f, %f], with confidence [%f]", frame_num, test.p[0], test.p[1], test.p[2], test.p[3], max_spt_conf);


    //do_sp_level_gauss_blur(test, pro_image_w1);

    cv::Mat backproj = pro_image_w1.clone();
    cv::Rect trackWindow(max_col + w / 2 - w / 8, max_row + h / 2 - h / 8, w / 4, h / 4);
    RotatedRect trackBox = CamShift(backproj, trackWindow,
            TermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1 ));

    if( trackWindow.area() <= 1 )
    {
        int cols = backproj.cols, rows = backproj.rows, r = (MIN(cols, rows) + 5)/6;
        trackWindow = Rect(trackWindow.x - r, trackWindow.y - r,
                trackWindow.x + r, trackWindow.y + r) &
            Rect(0, 0, cols, rows);
    }

    cv::Mat res = test.warpimg.clone();
    ellipse(res, trackBox, Scalar(0,0,255), 3, CV_AA );
    cv::rectangle(res, cv::Rect(max_col, max_row, test.p[2], test.p[3]), cv::Scalar(200, 0, 0));
    drawMapPoints(res, pTracker);
    imshow("track result", res);
    cv::waitKey(1);

    test.spt_conf = max_spt_conf;
    float occlusion_conf = (0.5 - min_prob) / (max_prob - min_prob) - max_spt_conf / test.p[2] / test.p[3];
    SLAM_DEBUG("[%d] frame, occlusion_conf [%f], threshold [%f]", frame_num, occlusion_conf, (occlusion_rate - min_prob)/ (max_prob - min_prob));
    if (occlusion_conf > (occlusion_rate - min_prob) / (max_prob - min_prob)){
        SLAM_DEBUG("severe occlustion");
        update_flag = false;
    } else {
        update_flag = true;
        test.warp_p = test.p;
        test.warp_p[0] = test.warp_p[0] - test.x1;
        test.warp_p[1] = test.warp_p[1] - test.y1;
    }

    // transform to 149*149 score;
    //sz = size(tmpl.mean);
    char buf[256];

    int res_rows = res.rows * 3;
    int res_cols = res.cols * 2;
    cv::Mat res_out = cv::Mat::zeros(res_rows, res_cols, CV_8UC3);

    int from_to[] =  {0, 0, 0, 1, 0, 2};
    cv::Mat prob3c(pro_image_w1_255.size(), CV_8UC3);
    mixChannels(&pro_image_w1_255, 1, &prob3c, 1, from_to, 3);
    prob3c.copyTo(res_out(cv::Rect(0, 0, res.cols, res.rows)));
    mixChannels(&pro_image_w1_ori_255, 1, &prob3c, 1, from_to, 3);
    prob3c.copyTo(res_out(cv::Rect(0, res.rows, res.cols, res.rows)));
    seg.copyTo(res_out(cv::Rect(res.cols, 0, res.cols, res.rows)));
    res.copyTo(res_out(cv::Rect(res.cols, res.rows, res.cols, res.rows)));
    test.imgWithContours.copyTo(res_out(cv::Rect(res.cols , res.rows * 2, res.cols, res.rows)));

    sprintf(buf, "merge%05d.jpg", frame_num);
    imwrite(buf, res_out);
}

void setProbs(cv::Mat& pro_image_w1, int *labels, int rows, int cols, vector<float>& w1){
    for(int row = 0; row < rows; ++row){
        for(int col = 0; col < cols; ++col){
            pro_image_w1.at<float>(row, col) = w1[*(labels + row * cols + col)];
        }
    }
}
void filterOutValue(cv::Mat& pro_image_w1, int rows, int cols, float threshold){
    for(int row = 0; row < rows; ++row){
        for(int col = 0; col < cols; ++col){
            if (pro_image_w1.at<float>(row, col) <= threshold + 1e-10)
                pro_image_w1.at<float>(row, col) = 0;
        }
    }
}

// convert hsi to 512bins;
void SPTracker::t1_cal_hsi_hist(SPTracker::SPTFrame& frame,int f, vector<pair<int, int>>&temp_index){

    auto ff = frame.warpimg_hsi;
    int cy = ff.rows;
    int cx = ff.cols;

    vector<cv::Mat> OriChannels(3);
    vector<cv::Mat> IntChannels(3);
    vector<cv::Mat> ResChannels(3);
    cv::split(ff, OriChannels);
    IntChannels[0] = OriChannels[0] * ch_bins_num - 0.5;
    IntChannels[0].convertTo(ResChannels[0], CV_16UC1);

    IntChannels[1] = OriChannels[1]  * ch_bins_num - 0.5;
    IntChannels[1].convertTo(ResChannels[1], CV_16UC1);
    ResChannels[1] = ResChannels[1] * ch_bins_num;

    IntChannels[2] = OriChannels[2] * ch_bins_num - 0.5;
    IntChannels[2].convertTo(ResChannels[2], CV_16UC1);
    ResChannels[2] = ResChannels[2] * pow(ch_bins_num, 2);
    //for(int row = 0; row < channels[0].rows; ++row){
    //printf("\n=====line %d=======\n", row);
    //for(int col = 0; col < channels[0].cols; ++ col){
    ////printf("[%d, %d] ", IntChannels[0].at<uint16_t>(row, col), f_bins.at<uint16_t>(row, col));
    //printf("[%u] ", IntChannels[0].at<uint16_t>(row, col));
    //}
    //}
    //abort();
    cv::Mat f_bins_tmp_1;
    cv::Mat f_bins_tmp_sum;
    cv::Mat f_bins;
    //printf("before weighted add\n");
    //for(int row = 0; row < channels[0].rows; ++row){
    //for(int col = 0; col < channels[0].cols; ++ col){
    //printf("[%f, %f, %f] ", channels[0].at<float>(row, col), channels[1].at<float>(row, col), channels[2].at<float>(row, col));
    //}
    //printf("\n");
    //abort();
    //}
    //
    addWeighted(ResChannels[0], 1, ResChannels[1], 1, 0, f_bins_tmp_1);
    addWeighted(ResChannels[2], 1, f_bins_tmp_1, 1 , 0, f_bins_tmp_sum);
    SLAM_DEBUG("after weighted add");

    //for(int row = 0; row < OriChannels[0].rows; ++row){
    //printf("\n=====line %d=======\n", row);
    //for(int col = 0; col < OriChannels[0].cols; ++ col){
    ////printf("[%d, %d] ", IntChannels[0].at<uint16_t>(row, col), f_bins.at<uint16_t>(row, col));
    //printf("[ %f %u, %f %u, %f %u, %f %u] ", OriChannels[0].at<float>(row, col), ResChannels[0].at<uint16_t>(row, col), OriChannels[1].at<float>(row, col), ResChannels[1].at<uint16_t>(row, col), OriChannels[2].at<float>(row, col), ResChannels[2].at<uint16_t>(row, col), f_bins_tmp_sum.at<uint16_t>(row, col));
    //}
    //}
    //abort();
    f_bins_tmp_sum.convertTo(f_bins, CV_16UC1);
    //float max0 = 0;
    //float max1 = 0;
    //float max2 = 0;
    //float maxn = 0;
    //uint16_t maxu = 0;
    //for(int row = 0; row < IntChannels[0].rows; ++row){
    //for(int col = 0; col < IntChannels[0].cols; ++ col){
    //max0 = max(max0, IntChannels[0].at<float>(row, col));
    //max1 = max(max1, IntChannels[1].at<float>(row, col));
    //max2 = max(max2, IntChannels[2].at<float>(row, col));
    //maxn = max(maxn, f_bins_tmp_sum.at<float>(row, col));
    //maxu = max(maxu, f_bins.at<uint16_t>(row, col));
    //printf("[%f, %f, %f; %f, %u] ", IntChannels[0].at<float>(row, col), IntChannels[1].at<float>(row, col), IntChannels[2].at<float>(row, col), f_bins_tmp_sum.at<float>(row, col), f_bins.at<uint16_t>(row, col));
    //}
    //printf("\n");
    //}
    //printf("\nmax val [%f, %f, %f; %f, %d]\n", max0, max1, max2, maxn, maxu);
    //abort();
    //

    //for(int row = 0; row < IntChannels[0].rows; ++row){
    //for(int col = 0; col < IntChannels[0].cols; ++ col){
    ////printf("[%d, %d] ", IntChannels[0].at<uint16_t>(row, col), f_bins.at<uint16_t>(row, col));
    //printf("[%d] ", f_bins.at<uint16_t>(row, col));
    //}
    //printf("\n");
    //}
    //abort();

    int* temp_labels = frame.labels;; //TODO use opencv mat;
    frame.sp_cl_hist.clear();
    frame.sp_cl_hist.resize(pow(ch_bins_num, 3), vector<float>(frame.sp_num, 0.0));

    vector<int> sp_pixel_num(frame.sp_num, 0);
    vector<vector<int>> sp_pixel_val(frame.sp_num, vector<int>());
    temp_index.resize(frame.sp_num, make_pair<int, int>(0, 0));


    int max_val = pow(ch_bins_num, 3) - 1;
    SLAM_DEBUG("stat pixel num and val");
    bool overflow = false;
    for(int row = 0; row < cy; ++row){
        for(int col = 0; col < cx; ++col){
            int sp_index = *(frame.labels + col + row * cx);
            int sp_bin_val = f_bins.at<uint16_t>(row, col);
            if (sp_bin_val < 0){
                printf("val : %d, row %d, col%d \n", sp_bin_val, row, col);
                continue;
            }
            if (sp_bin_val > max_val && !overflow){
                overflow = true;
                printf("sp_bin_val[%d] larger than max_val \n", sp_bin_val);
            }
            sp_pixel_num[sp_index] += 1;
            sp_pixel_val[sp_index].push_back(min(max_val, sp_bin_val));// calc row and col
        }
    }
    SLAM_DEBUG("after stat pixel num and pixel val");
    SLAM_DEBUG("size of sp_cl_hist [%d, %d]", frame.sp_cl_hist.size(), frame.sp_cl_hist[0].size());
    for(int spNo = 0; spNo < frame.sp_num; spNo++){
        if (sp_pixel_val[spNo].size() == 0){
            SLAM_DEBUG("****FATAL*** super pixel with no point ");
        }
        sort(sp_pixel_val[spNo].begin(), sp_pixel_val[spNo].end());
        vector<int> unique_value;
        vector<int> count;
        unique_value.push_back(sp_pixel_val[spNo][0]);
        count.push_back(1);
        for(int i = 1; i < sp_pixel_val[spNo].size(); ++i){
            if (sp_pixel_val[spNo][i] != unique_value.back()){
                unique_value.push_back(sp_pixel_val[spNo][i]);
                count.push_back(1);
            } else {
                count.back() += 1;
            }
        }
        //printf("stat spNo %d/ %d,size %d done\n", spNo, frame.sp_num, sp_pixel_val[spNo].size());
        float sum = 0;
        for(int idx = 0; idx < unique_value.size(); ++ idx){
            //printf("value %d, count %d\n", unique_value[idx], count[idx]);
            frame.sp_cl_hist[unique_value[idx]][spNo] = count[idx] / float(sp_pixel_val[spNo].size());
            sum += frame.sp_cl_hist[unique_value[idx]][spNo];
        }
        if (abs(sum - 1) > 1e-2){
            printf("sum value %f", sum);
        }
        sum = 0;
        for(int i = 0; i < 512; ++i){
            sum += frame.sp_cl_hist[i][spNo] ;
        }
        if (abs(sum - 1) > 1e-2){
            printf("sum value %f", sum);
        }
    }

    //check_sum("test just after calc histogram", frame.sp_cl_hist);
    //SLAM_DEBUG("test completed");
    SLAM_DEBUG("after cal hsi hisgogram, sp num %d", frame.sp_num);
    temp_index.resize(frame.sp_num);
    for(int spNo = 0; spNo < frame.sp_num; ++ spNo){
        temp_index[spNo].first = f;
        temp_index[spNo].second = spNo;
    }
    SLAM_DEBUG("end of func t1_cal_hsi_hist");
}

void SPTracker::t1_construct_appearance_model(vector<vector<float>>& train_sum_hist, int train_sp_sum_num){
    meanshiftCluster(train_sum_hist, cluster_bandWidth, false, clustCent, data2cluster, cluster2data);
    int cluster_sum = cluster2data.size();
    SLAM_DEBUG("[%d] clusters", cluster_sum);
    t1_show_cal_cluster_wt(train_frame_num, cluster_sum, data2cluster, train_frames, train_cluster_weight, train_index_pre);
    SLAM_DEBUG("appearance model constructed");

    SLAM_DEBUG("configure test");
    int f = train_frame_num - 1;
    test.block_size.first = train_frames[f].p[4];    
    test.block_size.second = train_frames[f].p[3];    
    test.p = train_frames[f].p;
    for(int i = 0; i < train_frames.size(); ++ i){
        SLAM_DEBUG("frame %d/%d,p value: [%f %f %f %f]", i, train_frame_num - 1, train_frames[i].p[0], train_frames[i].p[1], train_frames[i].p[2], train_frames[i].p[3]);
    }
    SLAM_DEBUG("test p value: [%f %f %f %f]", test.p[0], test.p[1], test.p[2], test.p[3]);
    //test.est = affparam2ultimate(test.p, test.block_size);
    test.warpimg = train_frames[f].warpimg;
    test.warp_p = train_frames[f].warp_p;
    test.train_sp_sum_num = train_sp_sum_num;
}

void SPTracker::t1_show_cal_cluster_wt(int frame_num, int cluster_sum, vector<int>& data2cluster, vector<SPTFrame>& frames, vector<pair<double, double>>& TrainCluster_Weight, vector<int>& sp_index_pre ){
    SLAM_DEBUG("%d clusters, do weight calculation", cluster_sum);
    TrainCluster_Weight.resize(cluster_sum, make_pair(0.0, 0.0));
    vector<vector<double>> temp_train_cluster_wt;
    temp_train_cluster_wt.resize(2, vector<double>(cluster_sum, 0.0));
    SLAM_DEBUG("===sp_index_pre:val");
    copy(sp_index_pre.begin(), sp_index_pre.end(), ostream_iterator<int>(cout, ","));
    printf("\n");
    for(int f = 0; f < frame_num; ++f){
        int* temp_labels = frames[f].labels;
        auto temp_warp_p = frames[f].warp_p;
        int cy = frames[f].warpimg_hsi.rows;
        int cx = frames[f].warpimg_hsi.cols;
        double costheta = cos(temp_warp_p[4]);
        double sintheta = sin(temp_warp_p[4]);
        int x0 = costheta * temp_warp_p[0] + sintheta * temp_warp_p[1];
        int y0 = costheta * temp_warp_p[1] + sintheta * temp_warp_p[0];
        printf("iterate through warpimg no %d with size [%d, %d], center[%d, %d]\n", f, cy, cx, x0, y0);

        for(int row = 0; row < cy; row++){
            for(int col = 0; col < cx; col++){
                int sp_no = (*(temp_labels + row * cx + col));
                //printf("%d sp no;\n", sp_no);
                int pixel_index = sp_index_pre[f] + sp_no;
                //printf("pixel index %d\n", pixel_index);
                if (pixel_index < 0 || pixel_index > 10000){
                    SLAM_FATAL("wrong pixel index with value %d, from %d %d in row col %d %d, relative to [%d, %d] with pointer [%p]", pixel_index, sp_index_pre[f], sp_no, row, col, cy, cx, temp_labels);// buggy TODO
                    for (row = 0; row < cy; row++){
                        for(col = 0; col < cx; col++){
                            int no = (*(temp_labels + row * cx + col));
                            printf("%d,", no );
                        }
                        printf("\n");
                    }
                    abort();
                    continue;
                }
                if (pixel_index >= data2cluster.size()){
                    pixel_index = data2cluster.size() - 1;
                }
                int temp_cluster_num = data2cluster[pixel_index];
                //printf("corresponding cluster num %d\n", temp_cluster_num);
                float x = costheta * col + sintheta * row;
                float y = costheta * row + sintheta * col;
                float half_w = 0.5 * temp_warp_p[2];
                float half_h = 0.5 * temp_warp_p[3];
                if ( (row *cx + col)% 37 == 0){
                    //printf("processing point [%f, %f] in sp [%d] relative to center [%d, %d] in region[%f, %f]\n", x, y, temp_cluster_num, x0, y0, half_w, half_h);
                    if (x0 < 0 || y0 < 0){
                        SLAM_DEBUG("error center location");
                        std::copy(temp_warp_p.begin(), temp_warp_p.end(), std::ostream_iterator<float>(std::cout, ", "));
                        SLAM_DEBUG("cos and sin %f %f", costheta, sintheta);
                        printf("\n");
                    }
                }
                if (temp_cluster_num < 0 || temp_cluster_num > cluster_sum){
                    SLAM_FATAL("fatal error: with invalid temp_cluster_num %d", temp_cluster_num);
                    abort();
                }
                if (pixel_index >= map_background.size()){
                    SLAM_DEBUG("**FATAL** pixel_index [%d], map_background.size() [%d]", pixel_index, int(map_background.size()));
                    //    map_background.resize(pixel_index, 0);
                }
                //if (map_background[pixel_index]){
                //    SLAM_DEBUG("false sp by map point [frame_num, sp_no, pixel_index], [%d, %d, %d]", frame_num - 14 + f, sp_no, pixel_index);
                //}
                if (abs(costheta * col + sintheta * row - x0) 
                        < (0.5 * temp_warp_p[2])
                        && abs(costheta * row + sintheta * col - y0) 
                        < (0.5 * temp_warp_p[3])){
                    if (map_background[pixel_index] == 0){
                        temp_train_cluster_wt[0][temp_cluster_num] += 1;
                    } else {
                        temp_train_cluster_wt[0][temp_cluster_num] += 1 - map_background[pixel_index];
                        temp_train_cluster_wt[1][temp_cluster_num] += map_background[pixel_index];
                    }
                } else {
                    temp_train_cluster_wt[1][temp_cluster_num] += 1;
                }
            }
        }
    }
    SLAM_DEBUG("stat foreground and background pixel num completed");
    SLAM_DEBUG("doing weight computation");
    int negative_sp_count = 0;
    int nan_sp_count = 0;
    for(int idx = 0; idx < cluster_sum; ++ idx){
        TrainCluster_Weight[idx].first = (temp_train_cluster_wt[0][idx] - temp_train_cluster_wt[1][idx]) /(temp_train_cluster_wt[0][idx] + temp_train_cluster_wt[1][idx]);
        TrainCluster_Weight[idx].second= (temp_train_cluster_wt[0][idx]) /(temp_train_cluster_wt[0][idx] + temp_train_cluster_wt[1][idx]);
        if (std::isnan((double)TrainCluster_Weight[idx].first)){
            nan_sp_count++;
            TrainCluster_Weight[idx].first = -1;
            SLAM_FATAL("not a number with value %lf %lf", temp_train_cluster_wt[0][idx], temp_train_cluster_wt[1][idx]);
            TrainCluster_Weight[idx].first = 0.0;
        }
        if (TrainCluster_Weight[idx].first < 0){
            negative_sp_count ++;
            TrainCluster_Weight[idx].first = TrainCluster_Weight[idx].first / negative_penality_ratio;
        }
        printf("%f ", TrainCluster_Weight[idx].first);
    }
    SLAM_DEBUG("weight calc completed [nan : negative :all] [%d, %d, %d]", nan_sp_count, negative_sp_count, cluster_sum);
}


void SPTracker::t1_update_info(){

    SLAM_DEBUG("collecting update info");
    SLAM_DEBUG("pre size %d, frames size [%d]", (int) update_index_pre.size(), int(update_frames.size()));
    if (update_flag == true){
        update_interval_num = update_interval_num + 1;
        if ( update_interval_num % update_spacing == 0){
            for(int i = 0; i < update_frames.size(); ++i){
                printf("%p, %d ", update_frames[i].labels, update_frames[i].instance_no);
            }
            printf("\n");
            if (update_index_pre.size() >= update_incre_num){
                update_frames.erase(update_frames.begin());
                SLAM_DEBUG("!!!**************new frame label pointer is %p", update_frames.back().labels);
                int outdatedcount = update_index_pre[0];
                for(int i = 0; i < update_index_pre.size(); ++i){
                    update_index_pre[i] -= outdatedcount;
                }
                update_index_pre.erase(update_index_pre.begin());
                eraseHist(update_hist_sum, outdatedcount);
                map_background.erase(map_background.begin(), map_background.begin() + outdatedcount);
                update_spt_conf.erase(update_spt_conf.begin());
            }
            update_num = update_num + 1;
            update_frames.push_back(SPTFrame());
            //std::copy(update_index_pre.begin(), update_index_pre.end(), std::ostream_iterator<int>(std::cout, " "));
            //not used incre_prob = incre_prob + test.spt_conf;
            update_frames.back().labels = test.labels;
            for(int i = 0; i < update_frames.size(); ++i){
                printf("[%p, %d] ", update_frames[i].labels, update_frames[i].instance_no);
            }
            printf("\n");
            test.labels = nullptr;
            update_frames.back().warpimg = test.warpimg;
            update_frames.back().warpimg_hsi = test.warpimg_hsi;
            update_frames.back().warp_p = test.warp_p;
            update_frames.back().spt_conf = test.spt_conf;
            update_frames.back().p = test.p;
            update_frames.back().map_background = test.map_background;
            update_frames.back().sp_num = test.sp_num;
            update_frames.back().adjacent_table.resize(test.sp_num);
            for(int i = 0; i < test.sp_num; ++i){
                update_frames.back().adjacent_table[i] = test.adjacent_table[i];
            }
            addHist(update_hist_sum, test.sp_cl_hist);
            
            copy(test.map_background.begin(), test.map_background.end(), back_inserter(map_background));
            update_spt_conf.push_back(test.spt_conf);
            //update_sp_num += test.sp_num;
            update_index_pre.push_back(update_index_pre.back() + test.sp_num);
            update_index_pre_final = update_index_pre;
            update_index_pre_final.insert(update_index_pre_final.begin(), 0);
            SLAM_DEBUG("update info collected");
        }
    } else {
        SLAM_DEBUG("FATAL: not implement for severe occlusion");
    }
    SLAM_DEBUG("update info collected");
}

void SPTracker::t1_update_app_model(){
    SLAM_DEBUG("updating appearance model");
    update_interval_num = 0;
    float save_prob = std::accumulate(update_spt_conf.begin(), update_spt_conf.end(), 0.0) / update_incre_num;
    vector<double> diff(update_spt_conf.size(), 0);
    std::transform(update_spt_conf.begin(), update_spt_conf.end(), diff.begin(), [save_prob](double x){ return x - save_prob;});
    float save_std= sqrt(std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0) / diff.size() );

    meanshiftCluster(update_hist_sum, cluster_bandWidth, false, clustCent, data2cluster, cluster2data);

    int cluster_sum = cluster2data.size();
    test.train_sp_sum_num = update_index_pre.back();
    t1_show_cal_cluster_wt(update_incre_num, cluster_sum, data2cluster, update_frames, train_cluster_weight, update_index_pre_final);
    train_sum_hist = update_hist_sum;
    update_num = 0;
    update_sp_num = 0;
    update_flag = 1;
    update_negative = 0;
    SLAM_DEBUG("update appearance model completed");
}

void SPTracker::meanshiftCluster(vector<vector<float>>& dataPts, float bandWidth, bool plotFlag, vector<vector<float>>& clustCent, vector<int>& data2cluster, vector<vector<int>>& cluster2data){
    int non_zero_count = 0;
    vector<float> pb_sum(dataPts[0].size(), 0);
    for(int dimNo = 0; dimNo < dataPts.size(); ++ dimNo){
        for(int ptIdx = 0; ptIdx < dataPts[0].size(); ++ptIdx){
            pb_sum[ptIdx] += dataPts[dimNo][ptIdx];
            if (dataPts[dimNo][ptIdx] != 0) {
                non_zero_count ++;
            }
        }
    }
    int non_one_count = 0;
    for(int ptIdx = 0; ptIdx < pb_sum.size(); ++ ptIdx){
        if (abs(pb_sum[ptIdx] - 1) > 1e-1) {
            non_one_count ++;
            printf("none one value %f", pb_sum[ptIdx]);
        }
    }
    SLAM_DEBUG("entering meanshift cluster with [%d] pts, [%d] nonzero dim, [%d] non one count", dataPts[0].size(), non_zero_count, non_one_count);
    int numDim = dataPts.size();
    int numPts = dataPts[0].size();
    float bandSq = pow(bandWidth, 2);

    int numClust = 0;

    clustCent.clear();
    clustCent.resize(numDim);
    vector<int >initPtInds(numPts, 0);
    iota(initPtInds.begin(), initPtInds.end(), 0);

    vector<float> maxPos(numDim, 0);
    vector<float> minPos(numDim, 0);
    vector<float> boundBox(numDim, 0);
    float sizeSpace = 0;
    for(int dim = 0; dim < numDim; ++dim){
        maxPos[dim] = *max_element(dataPts[dim].begin(), dataPts[dim].end()); 
        minPos[dim] = *min_element(dataPts[dim].begin(), dataPts[dim].end());
        boundBox[dim] = maxPos[dim] - minPos[dim];
    }
    for(int i = 0; i < boundBox.size(); ++i){
        sizeSpace += pow(boundBox[i], 2);
    }
    sizeSpace = sqrt(sizeSpace);

    float stopThresh = 1e-3 * bandWidth;
    vector<int> beenVisitedFlag(numPts, 0);
    int numInitPts = numPts;
    vector<vector<int>> clusterVotes;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);

    int iter = 0;
    while(numInitPts){
        ++iter;
        //printf("cluster %d iteration i\n", iter);
        int tempInd = min((int)ceil((numInitPts - 1e-6) * dis(gen)), (int)initPtInds.size() - 1);//TODO random number generate normal or int?
        int stInd = initPtInds[tempInd];
        vector<float> myMean(numDim, 0);
        for(int i = 0; i < numDim; ++i){//TODO change to copy function
            myMean[i] = dataPts[i][stInd];
        }
        vector<int> myMembers;
        vector<int> thisClusterVotes(numPts, 0);
        int cl_iter = 0;
        bool do_clust = true;
        while(do_clust){
            ++cl_iter;
            //printf("cluster %d iteration %d\n", iter, cl_iter);
            vector<float> sqDistToAll(numPts, 0);
            // square distance
            for(int dim = 0; dim < numDim; ++dim){
                for(int i = 0; i < numPts; ++i){
                    sqDistToAll[i] += pow((myMean[dim] - dataPts[dim][i]), 2);
                }
            }


            // add points within bandwidth to cluster
            int inPtsCount = 0;
            vector<float> myOldMean(numDim, 0);
            myMean.swap(myOldMean);
            for(int ptInd = 0; ptInd < sqDistToAll.size(); ++ptInd){
                if (sqDistToAll[ptInd] < bandSq){
                    thisClusterVotes[ptInd] += 1;
                    myMembers.push_back(ptInd);
                    beenVisitedFlag[ptInd] = 1;
                    inPtsCount ++;
                    for(int dim = 0; dim < numDim; ++ dim){
                        myMean[dim] += dataPts[dim][ptInd];
                    }
                }
            }

            for_each(myMean.begin(), myMean.end(), [inPtsCount](float& value){value/= inPtsCount;});

            float meanDist = 0;
            for(int i = 0; i < myMean.size(); ++i){
                meanDist += pow((myMean[i] - myOldMean[i]), 2);
            }
            meanDist = sqrt(meanDist);
            //printf("meanDist %f\n", meanDist);
            if (meanDist < stopThresh){
                // check for merge possibilities
                int mergeWith = -1;
                for(int clust = 0; clust < numClust; ++clust){
                    float distToOther = 0;
                    for(int dim = 0; dim < numDim; ++dim){
                        distToOther += pow((myMean[dim] - clustCent[dim][clust]), 2);
                    }
                    distToOther = sqrt(distToOther);
                    if (distToOther < bandWidth / 2){
                        mergeWith = clust;
                        break;
                    }
                }
                if (mergeWith >= 0){// merge
                    for(int dimNo = 0; dimNo < numDim; ++dimNo){
                        clustCent[dimNo][mergeWith] = 0.5 * (myMean[dimNo] + clustCent[dimNo][mergeWith]);// TODO modify biased toward new one?
                    }
                    for(int i = 0; i < thisClusterVotes.size(); ++i){
                        clusterVotes[mergeWith][i] += thisClusterVotes[i];
                    }

                } else { // new clust
                    //SLAM_DEBUG("new cluster center added");
                    //printf("[");
                    for(int dimNo =0; dimNo < numDim; ++dimNo){
                        //printf("%f, ", myMean[dimNo]);
                        clustCent[dimNo].push_back(myMean[dimNo]);
                    }
                    //printf("]\n");
                    clusterVotes.push_back(thisClusterVotes);
                    numClust++;
                }
                do_clust = false;

            }
        }

        vector<int> notVisited;
        for(int i = 0; i < beenVisitedFlag.size(); ++i){
            if (beenVisitedFlag[i] == 0){
                notVisited.push_back(i);
            }
        }
        initPtInds.swap(notVisited);
        numInitPts = initPtInds.size();
    }

    SLAM_DEBUG("cluster done with %d centers", numClust);
    vector<int> val(numPts, 0);
    data2cluster.clear();
    data2cluster.resize(numPts, -1);
    for(int clustIdx = 0; clustIdx < numClust; clustIdx++){
        for(int ptIdx = 0; ptIdx < numPts; ptIdx++){
            if (clusterVotes[clustIdx][ptIdx] > val[ptIdx]){
                val[ptIdx] = clusterVotes[clustIdx][ptIdx];
                data2cluster[ptIdx] = clustIdx;
            }
        }
    }

    SLAM_DEBUG("data2cluster done");
    SLAM_DEBUG("cluster2data.size()[%d] to [%d]", int(cluster2data.size()), numClust);
    //swap(cluster2data, vector<vector<int>>(numClust, vector<int>()));
    //cluster2data.clear();
    for(int i = 0; i < cluster2data.size(); ++i ){
        cluster2data[i].resize(0);
    }
    SLAM_DEBUG("clear succeed");
    cluster2data.resize(numClust);

    SLAM_DEBUG("cluster2data clear and resize done");
    for(int ptIdx = 0; ptIdx < numPts; ++ptIdx){
        //SLAM_DEBUG("adding point index %d to cluster %d", ptIdx, data2cluster[ptIdx]);
        cluster2data[data2cluster[ptIdx]].push_back(ptIdx);
    }

    //for(int ptIdx = 0; ptIdx < numPts; ++ ptIdx){
    //    if(map_background[ptIdx] == 1){
    //        for(int i = 0; i < cluster2data[data2cluster[ptIdx]].size() / 3; ++ i){
    //            map_background[cluster2data[data2cluster[ptIdx]][i]] = 1;
    //        }
    //    }
    //}
    SLAM_DEBUG("cluster2data done");

    SLAM_DEBUG("leaving meanshift cluster with [%d] clusters", cluster2data.size());
}

void SPTracker::addHist(vector<vector<float>>& hist_sum, vector<vector<float>>& hist){
    int numDim = pow(ch_bins_num, 3);
    int ptNum = hist_sum[0].size();
    for(int dimNo = 0; dimNo < numDim; ++ dimNo){
        copy(hist[dimNo].begin(), hist[dimNo].end(), back_inserter(hist_sum[dimNo]));
    }
}

void SPTracker::eraseHist(vector<vector<float>>& hist_sum, int count){
    int numDim = pow(ch_bins_num, 3);
    for(int dimNo = 0; dimNo < numDim; ++ dimNo){
        hist_sum[dimNo].erase(hist_sum[dimNo].begin(), hist_sum[dimNo].begin() + count);
    }
}

void rgb2hsi(cv::Mat& image, cv::Mat& res){
    cv::Mat ff;
    image.convertTo(ff, CV_32FC3, 1/ 255.0, 0);
    vector<cv::Mat> rgb;
    cv::split(ff, rgb);
    cv::Mat bm = rgb[0];
    cv::Mat gm = rgb[1];
    cv::Mat rm = rgb[2];
    res = cv::Mat(bm.rows, bm.cols, CV_32FC3);
    for(int row = 0; row < bm.rows; ++ row){
        for(int col = 0; col < bm.cols; ++col){
            float r = rm.at<float>(row, col);
            float g = gm.at<float>(row, col);
            float b = bm.at<float>(row, col);
            float num = 0.5*(r + r -g -b);
            float den = sqrt((r-g)*(r-g) + (r-b)*(g-b));
            float theta = acos(num/(den + 2.2204e-16));
            float H = theta;
            if (b > g){
                H = 2 * M_PI -H;
            }
            H = H / 2 / M_PI;
            num = min(min(r, g), b);
            den = r + g + b;
            float S = 0;
            if (den == 0) S = 1 ; // s = 1;
            else S = 1 - 3 * num / den;
            if (S == 0) H = 0;
            float I = (r+g+b)/3;
            res.at<cv::Vec3f>(row, col)[0] = H;
            res.at<cv::Vec3f>(row, col)[1] = S;
            res.at<cv::Vec3f>(row, col)[2] = I;
        }
    }

    bool debug = false;
    if (debug){
        SLAM_DEBUG("===== output raw and hsi values");
        for (int r = 0; r < image.rows; ++ r){
            for (int c = 0; c < image.cols; ++ c){
                cv::Vec3b p0 = image.at<cv::Vec3b>(r, c);
                cv::Vec3f p1 = ff.at<cv::Vec3f>(r, c);
                cv::Vec3f p2 = res.at<cv::Vec3f>(r, c);
                printf("([%d, %d, %d] [%f, %f, %f] [%f, %f, %f])", p0[2], p0[1], p0[0],p1[2], p1[1], p1[0], p2[0], p2[1], p2[2]);

            }
            printf("\n");
        }
    }
}


vector<double> SPTracker::affparam2mat(vector<double> param0){
    vector<double> result(6, 0.0);
    result[0] = param0[0];
    result[1] = param0[1];
    result[2] = param0[2];
    result[3] = 0;
    result[4] = 0;
    result[5] = param0[2] * param0[4];
}


void SPTracker::pointSetDistance(vector<vector<float>>& set1, vector<vector<float>>& set2, vector<vector<float>>& result){
    SLAM_DEBUG("entering pointSetDistance");
    if (set1.size() == 0 || set2.size() == 0){
        SLAM_DEBUG("one of the set has size 0");
        assert(false);
        return;
    }
    SLAM_DEBUG("dim of set1 [%d, %d], dim of set2[%d, %d]", set1.size(), set1[0].size(), set2.size(), set2[0].size());
    if (result.size() < set1[0].size() || result[0].size() < set2[0].size()){
        vector<vector<float>> res(set1[0].size(), vector<float>(set2[0].size()));
        result.swap(res);
    }
    SLAM_DEBUG("dim of result [%d, %d]", result.size(), result[0].size());
    //for(int idx1 = 0; idx1 < set1[0].size(); ++idx1){
    //    for(int idx2 = 0; idx2 < set2[0].size(); ++idx2){
    //        float dist = 0;
    //        for(int dimNo = 0; dimNo < set1.size(); dimNo ++){
    //            dist += pow((set1[dimNo][idx1] - set2[dimNo][idx2]), 2);
    //        }
    //        result[idx1][idx2] = dist;
    //    }
    //}
    //
    for (int dimNo = 0; dimNo < set1.size(); ++ dimNo){
        for(int idx1 = 0; idx1 < set1[0].size(); ++ idx1){
            for(int idx2 = 0; idx2 < set2[0].size(); ++idx2){
                result[idx1][idx2] += pow((set1[dimNo][idx1] - set2[dimNo][idx2]), 2);
            }
        }
    }
    SLAM_DEBUG("point distance calculation completed");
}


float SPTracker::pointDistanceSdColumn(vector<vector<float>>& p1, int colp1, vector<vector<float>>& p2, int colp2){
    float retval = 0;
    for(int dimNo = 0; dimNo < p1.size(); ++dimNo){
        retval += pow((p1[dimNo][colp1] - p2[dimNo][colp2]) , 2);
    }
    return retval;
}

float SPTracker::pointDistanceSd(vector<float>& p1, vector<float>& p2){
    assert(p1.size() == p2.size());
    float retval = 0;
    for(int dimNo = 0; dimNo < p1.size(); ++dimNo){
        retval += pow((p1[dimNo]  - p2[dimNo]) , 2);
    }
    return retval;
}
void sample_rect(std::string header, cv::Mat input){

}

void check_sum(std::string title, std::vector<std::vector<float>>& sp_cl_hist){
    int numDim = 512;
    printf("%s\n", title.c_str());
    for (int ptIdx = 0; ptIdx < sp_cl_hist[0].size(); ++ ptIdx){
        printf ("[%d, %d]", ptIdx, int(sp_cl_hist[0].size()));
        float sum = 0;
        for(int dimNo = 0; dimNo < numDim; ++ dimNo){
            sum += sp_cl_hist[dimNo][ptIdx];
        }
        if (abs(sum - 1) > 1e-1){
            printf("invalid sum %f\n", sum);
        }
    }
}

void SPTracker::utilizeMapPoints(cv::Mat& im, int x1, int y1, float fill_value, SPTFrame& frame){

    auto start_utilizemappoints = chrono::steady_clock::now();

    mark_background_and_set_prob(im, fill_value, frame);
    frame.distribute_map_background();

    auto end_utilizemappoints = chrono::steady_clock::now();
    auto diff = end_utilizemappoints- start_utilizemappoints;
    std::cout << "utilizemappoints consumes"<< chrono::duration <double, milli> (diff).count() << " ms" << std::endl;
}

void SPTracker::mark_background_and_set_prob(cv::Mat& im, float _fill_value, SPTFrame& frame){
    mnTracked = 0;
    mnTrackedVO = 0;
    mnMatchesBoth = 0;

    std::set<float> vals;
    std::set<int> sps;
    for(int i=0;i<N;i++)
    {
        if(mvbVO[i] || mvbMap[i])
        {
            bool m1 = mvbMapPointsMatchFromPreviousFrame[i];
            bool m2 = mvbMapPointsMatchFromLocalMap[i];
            if (m1 && m2){
                mnMatchesBoth++;
            }
            cv::Point2f pt1,pt2;
            //SLAM_DEBUG("flood filling point [%f, %f], origin[%d, %d]", mvCurrentKeys[i].pt.x,mvCurrentKeys[i].pt.y, x1, y1);
            //pt1.x=mvCurrentKeys[i].pt.x-r;
            //pt1.y=mvCurrentKeys[i].pt.y-r;
            //pt2.x=mvCurrentKeys[i].pt.x+r;
            //pt2.y=mvCurrentKeys[i].pt.y+r;

            // This is a match to a MapPoint in the map
            if(mvbMap[i])
            {

                if (mvbMapPointsMatchFromLocalMap[i]){
                    if (mvbMapPointsMatchFromPreviousFrame[i]){
                        mnMatchesBoth++;
                        find_vals_and_sps(im,mvCurrentKeys[i].pt.x, mvCurrentKeys[i].pt.y, _fill_value, frame, vals, sps);
                        //cv::rectangle(im,pt1,pt2,cv::Scalar(0,0, 200));
                        //cv::circle(im,mvCurrentKeys[i].pt,2,cv::Scalar(0, 0, 200),-1);
                    } else {
                        find_vals_and_sps(im,mvCurrentKeys[i].pt.x, mvCurrentKeys[i].pt.y, _fill_value, frame, vals, sps);
                        //cv::rectangle(im,pt1,pt2,cv::Scalar(200,0, 0));
                        //cv::circle(im,mvCurrentKeys[i].pt,2,cv::Scalar(200,0,0),-1);
                    }

                } else {
                    find_vals_and_sps(im,mvCurrentKeys[i].pt.x, mvCurrentKeys[i].pt.y, _fill_value, frame, vals, sps);
                    //cv::rectangle(im,pt1,pt2,cv::Scalar(0,200,0));
                    //cv::circle(im,mvCurrentKeys[i].pt,2,cv::Scalar(0,200,0),-1);

                }
                mnTracked++;
            }
            else // This is match to a "visual odometry" MapPoint created in the last frame
            {
                //cv::rectangle(im,pt1,pt2,cv::Scalar(255,0,0));
                //cv::circle(im,mvCurrentKeys[i].pt,2,cv::Scalar(255,0,0),-1);
                mnTrackedVO++;
            }
        }
    }
    for(auto iter = sps.begin(); iter != sps.end(); ++iter){
        frame.map_background[*iter] = 1;
    }
    fill_value(im, _fill_value, vals, sps);
}

int get_sp_label(cv::Mat& im, int x, int y, SPTracker::SPTFrame& frame){
    if (x >=0 && x < im.cols && y >= 0 && y < im.rows){
        return *(frame.labels + y*frame.warpimg_hsi.cols + x);
    } else {
        return -1;
    }
}

void find_vals_and_sps(cv::Mat& im, int x, int y, float _fill_value, SPTracker::SPTFrame& frame, std::set<float>& vals, std::set<int>& sps){
    //SLAM_DEBUG("flood fill point [%d, %d], find vals", x, y);
    if (x < 0 || y < 0 || x >= im.cols || y >= im.rows){
        return;
    }

    for(int i = -4; i <= 4; ++ i){
        for(int j = -4; j <= 4; ++j){
            int label = get_sp_label(im, x + i, y + j, frame);
            if (label >= 0){
                vals.insert(im.at<float>(y+j, x+i)); 
                sps.insert(label);
            }
        }
    }
}

void fill_value(cv::Mat& im, float _fill_value, std::set<float>& vals, std::set<int>& sps){
    for(int r = 0; r < im.rows; ++r){
        for(int c = 0; c < im.cols; ++ c){
            if (vals.count(im.at<float>(r, c))){
                im.at<float>(r, c) = _fill_value;
            }
        }
    }
}


void floodFill(cv::Mat& im, int x, int y, std::set<float>& val){
    if (x < 0 || x >= im.cols || y < 0 || y >= im.rows){
        return;
    }
    if (im.at<float>(y, x) == -0.1111111){
        return ;
    }
    if (val.count(im.at<float>(y, x)) > 0){
        im.at<float>(y, x) = -0.1111111;//TODO fix dangerous magic value
        if (x - 1 >= 0){
            if (y - 1 >= 0)
                floodFill(im, y - 1, x - 1, val) ;
            if (y + 1 < im.rows)
                floodFill(im, y + 1, x - 1, val) ;
        }
        if (x + 1 < im.cols){
            if (y - 1 >= 0)
                floodFill(im, y - 1, x + 1, val) ;
            if (y + 1 < im.rows)
                floodFill(im, y + 1, x + 1, val) ;
        }
    } else {
        return;
    }

}

void SPTracker::drawMapPoints(cv::Mat& im, ORB_SLAM2::Tracking* pTracker){
    const float r = 5;
    mnTracked = 0;
    mnTrackedVO = 0;
    mnMatchesBoth = 0;
    char buf[256];
    int sampleCount = 0;
    for(int i=0;i<N;i++)
    {
        if(mvbVO[i] || mvbMap[i])
        {
            bool m1 = mvbMapPointsMatchFromPreviousFrame[i];
            bool m2 = mvbMapPointsMatchFromLocalMap[i];
            if (m1 && m2){
                mnMatchesBoth++;
            }
            cv::Point2f pt1,pt2;
            pt1.x=mvCurrentKeys[i].pt.x-r;
            pt1.y=mvCurrentKeys[i].pt.y-r;
            pt2.x=mvCurrentKeys[i].pt.x+r;
            pt2.y=mvCurrentKeys[i].pt.y+r;

            // This is a match to a MapPoint in the map
            if(mvbMap[i])
            {

                sprintf(buf," %.2f", depth[i]);
                if (mvbMapPointsMatchFromLocalMap[i]){
                    if (mvbMapPointsMatchFromPreviousFrame[i]){
                        mnMatchesBoth++;
                        SLAM_DEBUG("*********** from pre frame");
                        cv::rectangle(im,pt1,pt2,cv::Scalar(0,0, 200));
                        cv::circle(im,mvCurrentKeys[i].pt,2,cv::Scalar(0, 0, 200),-1);
                    } else {
                        cv::rectangle(im,pt1,pt2,cv::Scalar(200,0, 0));
                        cv::circle(im,mvCurrentKeys[i].pt,2,cv::Scalar(200,0,0),-1);
                        //cv::putText(im, buf,mvCurrentKeys[i].pt,cv::FONT_HERSHEY_PLAIN,0.7,cv::Scalar(0,0,200),1,8);
                    }

                } else {
                    cv::rectangle(im,pt1,pt2,cv::Scalar(0,200,0));
                    sampleCount++;
                    if (sampleCount % 4 == 0)
                        cv::putText(im, buf,mvCurrentKeys[i].pt,cv::FONT_HERSHEY_PLAIN,0.6,cv::Scalar(0,0,230),1,8);
                    else
                        cv::circle(im,mvCurrentKeys[i].pt,2,cv::Scalar(0,200,0),-1);


                }
                mnTracked++;
            }
            else // This is match to a "visual odometry" MapPoint created in the last frame
            {
                SLAM_DEBUG("***************vo point");
                //cv::rectangle(im,pt1,pt2,cv::Scalar(255,0,0));
                cv::circle(im,mvCurrentKeys[i].pt,2,cv::Scalar(255,0,0),-1);
                mnTrackedVO++;
            }
        }
    }
}

void SPTracker::copyMapPoints(ORB_SLAM2::Tracking* pTracker){

    mvCurrentKeys=pTracker->mCurrentFrame.mvKeys;
    N = mvCurrentKeys.size();
    depth.clear();
    depth.resize(N, 0);
    observation.clear();
    observation.resize(N, 0);
    SLAM_DEBUG("[%d] keypoints", N);
    mvbMapPointsMatchFromLocalMap = pTracker->mCurrentFrame.mvbMapPointsMatchFromLocalMap;
    mvbMapPointsMatchFromPreviousFrame = pTracker->mCurrentFrame.mvbMapPointsMatchFromPreviousFrame;

    mvbVO = vector<int>(N,false);
    mvbMap = vector<int>(N,false);
    mbOnlyTracking = pTracker->mbOnlyTracking;


    const cv::Mat Rcw = pTracker->mCurrentFrame.mTcw.rowRange(0,3).colRange(0,3);
    const cv::Mat tcw = pTracker->mCurrentFrame.mTcw.rowRange(0,3).col(3);


    vector<double>  times;
    for(int i=0;i<N;i++)
    {


        ORB_SLAM2::MapPoint* pMP = pTracker->mCurrentFrame.mvpMapPoints[i];
        if(pMP)
        {
            if(!pTracker->mCurrentFrame.mvbOutlier[i])
            {


                if(pMP->Observations()>0){
                    cv::Mat x3Dw = pMP->GetWorldPos();
                    cv::Mat x3Dc = Rcw*x3Dw+tcw;
                    //SLAM_DEBUG("****3dc cord [%f, %f, %f], ratio[%f]", x3Dc.at<float>(0),  x3Dc.at<float>(1), x3Dc.at<float>(2), pMP->GetFoundRatio());
                    depth[i] = x3Dc.at<float>(2);
                    observation[i] = pMP->Observations();

                    if(pMP->Observations()>=10){
                        times.push_back(pMP->Observations());
                        mvbMap[i]=true;
                    }
                }
                //else
                //    mvbVO[i]=true;
            }
        }
    }

    float min_times = *min_element(times.begin(), times.end());
    float max_times = *max_element(times.begin(), times.end());
    float mean= std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    vector<double> diff(times.size(), 0);
    std::transform(times.begin(), times.end(), diff.begin(), [mean](double x){ return x - mean;});
    float save_std= sqrt(std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0) / diff.size() );
    SLAM_DEBUG("***********min, max, mean, std [%f, %f %f, %f]", min_times, max_times, mean, save_std);

    return;
}


void SPTracker::dump_rect(int x, int y, int w, int h){
    if (!ofile.is_open()){
        ofile.open("./track_res.txt");
    }
    ofile<< x << " " << y << " " << w << " " << h <<std::endl;
}

void SPTracker::SPTFrame::generate_adjacent_table(){
    adjacent_table.clear();
    adjacent_table.resize(sp_num);
    int pre_label;
    SLAM_DEBUG("in generate adjacent table with [%d, %d] sp_num[%d]", warpimg_hsi.rows, warpimg_hsi.cols, sp_num);
    for (int r = 0; r < warpimg_hsi.rows; ++ r){
        for(int c = 0; c < warpimg_hsi.cols; ++c){
            int current_label = *(labels + r * warpimg_hsi.cols + c);
            if (c == 0){
                pre_label = current_label;
            }else{
                if (pre_label != current_label){
                    //SLAM_DEBUG("find neighbot %d %d, r, c, %d, %d", pre_label, current_label, r, c);
                    adjacent_table[pre_label].insert(current_label);
                    adjacent_table[current_label].insert(pre_label);
                    pre_label = current_label;
                }
            }
        }
    }

    vector<int> times;
    for(int i = 0; i < adjacent_table.size(); ++i){
        times.push_back(adjacent_table[i].size());
    }
    float min_times = *min_element(times.begin(), times.end());
    float max_times = *max_element(times.begin(), times.end());
    float mean= std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    vector<double> diff(times.size(), 0);
    std::transform(times.begin(), times.end(), diff.begin(), [mean](double x){ return x - mean;});
    float save_std= sqrt(std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0) / diff.size() );
    SLAM_DEBUG("*******neighbor****min, max, mean, std [%f, %f %f, %f]", min_times, max_times, mean, save_std);
}

void SPTracker::SPTFrame::distribute_map_background(){
    set<int> processed_sp;
    auto tmp_map_background = map_background;
    for(int i =0; i < sp_num; ++i){
        //SLAM_DEBUG("checking sp [%d/%d], map size[%d]", i, sp_num, map_background.size());
        if (tmp_map_background[i] == 0){
            continue;
        }
        for(auto iter = adjacent_table[i].begin(); iter != adjacent_table[i].end(); ++iter){
            //SLAM_DEBUG("distribute to %d", *iter);
            map_background[*iter] += 0.5;
        }
    }
}

int SPTracker::SPTFrame::prepare_info(float grid_ratio, int SLIC_sp_num){
    float w = p[2];
    float h = p[3];
    int temp_length = int(sqrt(w * w / 4 + h * h / 4) * grid_ratio);
    //TODO warn minus ceil, add floor
    x1 = max(0, int(p[0] - temp_length));
    y1 = max(0, int(p[1] - temp_length));
    x2 = min(image.cols - 1, int(p[0] + temp_length));
    y2 = min(image.rows - 1, int(p[1] + temp_length));
    int roi_w = x2 - x1 + 1;
    int roi_h = y2 - y1 + 1;
    SLAM_DEBUG("roi region [%d, %d; %d, %d] in image with size [%d, %d]", x1, y1, x2, y2, image.rows, image.cols);
    SLAM_DEBUG("roi computation completed with [x1, y1, x2, y2]:[%d, %d, %d, %d],"
            " [x,  y, w, h],[%d, %d, %d, %d]", x1, y1, x2, y2, x1, y1, roi_w, roi_h);

    warp_p = p;
    warp_p[0] = warp_p[0] - x1;
    warp_p[1] = warp_p[1] - y1;
    //train_frames[i].warpimg_tmpl.cx = train_frames[i].warpimg.width;
    //train_frames[i].warpimg_tmpl.cy = train_frames[i].warpimg.height;
    //train_frames[i].warp_param.est = affparam2ultimate(train_framtes[i].warp_p, opt.tmplsize);
    //
    warpimg = cv::Mat(image, cv::Rect(x1, y1, x2 - x1 + 1,  y2 - y1 + 1));
    SLAM_DEBUG("warp imagesize [%d, %d]", warpimg.rows, warpimg.cols);
    rgb2hsi(warpimg, warpimg_hsi);
    //SLAM_DEBUG("%d image type, %d image depth", train_frames[i].warpimg_hsi.type(), train_frames[i].warpimg_hsi.depth());

    SLIC slic;
    slic.GenerateSuperpixels(warpimg, SLIC_sp_num);
    labels = (int*)malloc(sizeof(int) * (warpimg_hsi.cols) * (warpimg_hsi.rows));
    imgWithContours = slic.GetImgWithContours(cv::Scalar(0, 0, 255));
    memcpy(labels, slic.GetLabel(), (warpimg_hsi.cols)*(warpimg_hsi.rows)*sizeof(int));
    vector<int> sp_no;
    int N_superpixels = statSuperpixel(labels, x2 - x1 + 1,  y2 - y1 + 1, sp_no);
    SLAM_DEBUG("N_superpixels %d", N_superpixels);
    sp_num = N_superpixels;
    map_background.clear();
    map_background.resize(N_superpixels, 0);
    generate_adjacent_table();
    return N_superpixels;
}

void find_min_sp_index_and_max_distance(vector<int>& temp_nearest_train_sp_index, vector<float>& temp_fatherest_train_sp_distance, vector<vector<float>>& cluster_dis){
    for(int i = 0; i < cluster_dis.size(); ++i){
        int min_index = 0;
        double min_dist = cluster_dis[i][0];
        double max_dist = cluster_dis[i][0];
        for(int j = 1; j < cluster_dis[i].size(); ++j){
            if (cluster_dis[i][j] < min_dist){
                min_index = j;
                min_dist = cluster_dis[i][j];
            } else {
                if (cluster_dis[i][j] > max_dist){
                    max_dist = cluster_dis[i][j];
                }
            }
        }
        temp_nearest_train_sp_index.push_back(min_index);
        temp_fatherest_train_sp_distance.push_back(max_dist);
    }
    SLAM_DEBUG("min dist index and max dist calculation completed");

}

void segByProb(cv::Mat& pro_image_w1_255, cv::Mat& seg){
    cv::Mat prob_bw;
    threshold(pro_image_w1_255, prob_bw, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
    for (int r = 0; r < prob_bw.rows; ++r){
        for(int c = 0; c < prob_bw.cols; ++ c){
            if (prob_bw.at<char>(r, c) == 0){
                seg.at<Vec3b>(r, c) = Vec3b(0, 0, 0);
            }
        }
    }
}
void find_max_spt_conf_location(cv::Mat& pro_image_w1, int w, int h, int& max_row, int& max_col, double& max_spt_conf){
    cv::Mat pro_image_w1_integral;
    integral(pro_image_w1, pro_image_w1_integral, CV_32F);
    SLAM_DEBUG("size of two image [%d, %d], [%d, %d]", pro_image_w1.rows, pro_image_w1.cols, pro_image_w1_integral.rows, pro_image_w1_integral.cols);
    max_spt_conf = -INT_MAX;
    double min_spt_conf = INT_MAX;
    max_row = 0;
    max_col = 0;
    for (int row = 0; row + h - 1 < pro_image_w1.rows; ++row){
        int ymin = row;
        int ymax = row + h - 1;
        for(int col = 0; col + w - 1 < pro_image_w1.cols; ++ col){
            int xmin = col;
            int xmax = col + w - 1;
            double val = pro_image_w1_integral.at<float>(ymax + 1, xmax + 1)
                - pro_image_w1_integral.at<float>(ymin, xmax + 1) 
                - pro_image_w1_integral.at<float>(ymax + 1, xmin) 
                + pro_image_w1_integral.at<float>(ymin, xmin);
            if (val > max_spt_conf){
                max_spt_conf = val;
                max_row =  row;
                max_col = col;
            }
            if (val < min_spt_conf){
                min_spt_conf = val;
            }
        }
    }
}

void find_weight(vector<float>& w2, cv::Mat& pro_image_w1, SPTracker::SPTFrame& test){
    int rows = test.warpimg_hsi.rows;
    int cols = test.warpimg_hsi.cols;
    for(int row = 0; row < rows; ++row){
        for(int col = 0; col < cols; ++col){
            int sp_no = *(test.labels + row * cols + col);
            w2[sp_no] = pro_image_w1.at<float>(row, col);
        }
    }

}
void sp_gauss_blur(vector<float>& w2, std::vector<std::set<int>>& adjacent_table, int level){
    auto w2_ori = w2;
    int sp_num = w2.size();
    for(int i =0; i < sp_num; ++i){
            w2[i] = w2[i]*0.5;
        for(auto iter = adjacent_table[i].begin(); iter != adjacent_table[i].end(); ++iter){
            //SLAM_DEBUG("distribute to %d", *iter);
            w2[i] += 0.1 * w2_ori[*iter];
        }
        if (w2[i] >= 1.0){
            w2[i] = 0.9999;
        }
    }
}
//TODO transform to cv::mat
