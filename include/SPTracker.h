#pragma once
#include <vector>
#include <utility>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <array>
#include "log.h"
#include "Tracking.h"
#include <iostream>
#include <fstream>
//#include "MapPoint.h"
using std::array;
using std::vector;
using std::pair;
using std::string;

//class ORB_SLAM2::Tracking;
class SPTracker{
public:
    SPTracker(void);
    ~SPTracker(void);
    void init(cv::Mat& frame, cv::Rect bb);
    void addTrainFrame(cv::Mat frame, cv::Rect newone);
    void train();
    void run(cv::Mat& frame, cv::Rect& new_location, ORB_SLAM2::Tracking* pTracker = nullptr);
    void do_tracking(cv::Mat& frame, ORB_SLAM2::Tracking* pTracker);

public:

    int frame_num;

    array<double, 6> p;
    // train option part
    // this part init on initialization
    //
    int grid_size;
    float grid_ratio;
    int train_frame_num;
    int show_patch_num;//?? for what
    int train_sp_sum_num;
    vector<vector<float>> train_sum_hist;
    vector<pair<int, int>> train_sum_hist_index;
    int framu_num;//??
    int offline_incre_num;
    vector<int>train_index_pre;
    // myopt ?? what is the difference case-irrelative??
    bool HSI_flag;
    int ch_bins_num;
    float cluster_bandWidth;
    int SLIC_spatial_proximity_weight;
    int SLIC_sp_num;
    double negative_penality_ratio = 3;
    float sigma_c;
    float sigma_s;
    // this part init after frame passed in
    double mot_sig_v1;
    double mot_sig_v2;
    pair<int, int> image_size;
    //int lambda_r;
    // end of myopt part;
   // ***opt part?? make any senses?? case relative??
    int numsample;
    vector<int> affsig;
    pair<int, int> tmplsize;
    double condenssig;// 0.2500
    vector<int> param0;
    //what this part for
    int maxbasis;
    int bastchsize;
    string errfunc;//"L2"
    double ff;//1.0


    float occlusion_rate;
    int update_incre_num;
    int update_frequency;
    int update_spacing;
    bool update_flag;
    int update_sp_sum_num;
    int update_num;
    bool update_negative;

    //   *** end of opt part 


    ofstream ofile;
    int numDim;

    //* train part;
    vector<cv::Rect> groundtruth_boxes;

    struct SPTFrame{
        static int SPTFrame_instance_no;
        int instance_no;
        SPTFrame(){
            labels = nullptr;
            train_sp_sum_num = 0;
            p[0] = 0;
            p[1] = 0;
            p[2] = 0;
            p[3] = 0;
            p[4] = 0;
            p[5] = 0;
            warp_p = p;
            sp_num = 0;
            train_sp_sum_num = 0;
            spt_conf = 0;
            SPTFrame::SPTFrame_instance_no++;
            instance_no = SPTFrame::SPTFrame_instance_no;
            SLAM_DEBUG("new SPTFrame [%d] created pointer", instance_no);
        }
        ~SPTFrame(){
            if (labels != nullptr){
                SLAM_DEBUG("deleting label pointer %p, instance_no[%d] ", labels, instance_no);
                delete labels;
            } else {
                SLAM_DEBUG("deleting nullptr pointer ?? instance_no[%d]", instance_no);
            }
        }
        SPTFrame(const SPTFrame& rhs){
            SPTFrame::SPTFrame_instance_no++;
            instance_no = SPTFrame::SPTFrame_instance_no;
            SLAM_DEBUG("new SPTFrame [%d] cloned pointer", instance_no);
            labels = nullptr;
            if (rhs.labels != nullptr){
                int size = sizeof(int) * rhs.warpimg_hsi.rows * rhs.warpimg_hsi.cols;
                labels = (int*)malloc(size);
                SLAM_DEBUG("!!!!!copying labels with size [%d, %d]!! from pointer [%p] to [%p], instance [%d] to [%d]", rhs.warpimg_hsi.rows, rhs.warpimg_hsi.cols, rhs.labels, labels, rhs.instance_no, instance_no);
                memcpy(labels, rhs.labels, size);
            }
            image = rhs.image;
            warpimg = rhs.warpimg;
            warpimg_hsi = rhs.warpimg_hsi;
            x1 = rhs.x1;
            x2 = rhs.x2;
            y1 = rhs.y1;
            y2 = rhs.y2;
            p = rhs.p;
            warp_p = rhs.warp_p;
            map_background = rhs.map_background;
            sp_num = rhs.sp_num;
            sp_cl_hist = rhs.sp_cl_hist;
            adjacent_table.resize(rhs.adjacent_table.size());
            for(int i = 0; i < adjacent_table.size(); ++i ){
                adjacent_table[i] = rhs.adjacent_table[i];
            }
            block_size = rhs.block_size;
            train_sp_sum_num = rhs.train_sp_sum_num;
            spt_conf = rhs.spt_conf;
            imgWithContours = rhs.imgWithContours;
        }
        SPTFrame& operator=(SPTFrame&& rhs){
            SLAM_DEBUG("copying from [%p, %d] to [%p, %d]", rhs.labels, rhs.instance_no, labels, instance_no);
            if (labels != nullptr){
                delete labels; 
                labels = nullptr;
            }
            labels = rhs.labels;
            rhs.labels = nullptr;
            image = rhs.image;
            warpimg = rhs.warpimg;
            warpimg_hsi = rhs.warpimg_hsi;
            x1 = rhs.x1;
            x2 = rhs.x2;
            y1 = rhs.y1;
            y2 = rhs.y2;
            p = rhs.p;
            warp_p = rhs.warp_p;
            map_background = rhs.map_background;
            sp_num = rhs.sp_num;
            sp_cl_hist = rhs.sp_cl_hist;
            adjacent_table.resize(rhs.adjacent_table.size());
            for(int i = 0; i < adjacent_table.size(); ++i ){
                adjacent_table[i] = rhs.adjacent_table[i];
            }
            block_size = rhs.block_size;
            train_sp_sum_num = rhs.train_sp_sum_num;
            spt_conf = rhs.spt_conf;
            imgWithContours = rhs.imgWithContours;
        }
        void generate_adjacent_table();
        void distribute_map_background();
        int prepare_info(float grid_ratio, int SLIC_sp_num);
        cv::Rect to_tl(){
            return cv::Rect(p[0] - p[2]/2, p[1] - p[3]/2, p[2], p[3]); 
        }

        cv::Mat image;
        cv::Mat warpimg;
        cv::Mat /*typ*/ warpimg_hsi;
        int x1, x2, y1, y2;
        array<float, 6> p;
        //vector<float> train_box;TODO when used
        array<float, 6> warp_p;
        vector<float> map_background;
        int* labels;
        int sp_num;
        vector<vector<float>> sp_cl_hist;
        std::vector<std::set<int>> adjacent_table;
        cv::Mat imgWithContours;
        //warpimg_tmpl; just recode w and h for hat? 149*149
        // test owned
        pair<int, int> block_size;
       // est;
        int train_sp_sum_num;
        float spt_conf;
    };

    vector<SPTFrame> train_frames;
    vector<pair<double, double>> train_cluster_weight;


    // model part
    vector<vector<float>> clustCent;
    vector<int> data2cluster;
    vector<vector<int>> cluster2data;
    int cluster_sum;

    // update part
    vector<SPTFrame> update_frames;
    vector<vector<float>> update_hist_sum ;
    int update_sp_num;
    vector<int> update_index_pre;
    vector<int> update_index_pre_final;
    vector<float> update_spt_conf;
    vector<float> map_background;
    
    // test part;
    SPTFrame test;
    cv::Rect last_box;
    int update_interval_num;//4

private:

    void calc_sp_w1(vector<int> &temp_nearest_train_sp_index, vector<float>& temp_fatherest_train_sp_distance, SPTFrame& frame, vector<float>& w1);
    void pointSetDistance(vector<vector<float>>& set1, vector<vector<float>>& set2, vector<vector<float>>& result);
    float pointDistanceSdColumn(vector<vector<float>>& p1, int colp1, vector<vector<float>>& p2, int colp2);
    float pointDistanceSd(vector<float>& p1, vector<float>& p2);
    //int statSuperpixel(int*, int w, int h, vector<int>& sp_no);
    void t1_cal_hsi_hist(SPTFrame& frame, int , vector<pair<int, int>>& temp_index);
    void meanshiftCluster(vector<vector<float>>& dataPts, float bandWidth, bool plotFlag, vector<vector<float>>& clustCent, vector<int>& data2cluster, vector<vector<int>>& cluster2data);
    void t1_construct_appearance_model(vector<vector<float>>& train_sum_hist, int train_sp_sum_num);
    void t1_show_cal_cluster_wt(int frame_num, int cluster_sum, vector<int>& data2cluster, vector<SPTFrame>& frames, vector<pair<double, double>>& TrainCluster_Weight, vector<int>& sp_index_pre);
    vector<double> affparam2mat(vector<double> param0);
    void addHist(vector<vector<float>>& hist_sum, vector<vector<float>>& hist);
    void eraseHist(vector<vector<float>>& hist_sum, int count);
    void t1_update_app_model();
    void t1_update_info();
    //void rgb2hsi(cv::Mat& image, cv::Mat& res);
    void copyMapPoints(ORB_SLAM2::Tracking *pTracker);
    void dump_rect(int x, int y, int w, int h);
    int img_width;
    int img_height;


private:
    // SLAM keypoint mappoint related variable
    vector<cv::KeyPoint> mvCurrentKeys;
    vector<int> mvbMap, mvbVO;
    int N;

    bool mbOnlyTracking;

    vector<bool> mvbMapPointsMatchFromLocalMap;
    vector<bool> mvbMapPointsMatchFromPreviousFrame;
    int mnTracked, mnTrackedVO;
    int mnMatchesBoth;
    void drawMapPoints(cv::Mat& im, ORB_SLAM2::Tracking* pTracker);
    void utilizeMapPoints(cv::Mat& prob, int x1, int y1, float fill_value, SPTFrame& frame);
    void mark_background_and_set_prob(cv::Mat& im, float _fill_value, SPTFrame& frame);
    vector<float> depth;
    vector<int> observation;

};
