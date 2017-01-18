#pragma once
#define LOGLINE(line) LOGLINE_(line)
#define LOGLINE_(line) #line
#define SLAM_DEBUG(fmt, ...) do{\
    printf("[" __FILE__ ":" LOGLINE(__LINE__) "]  " fmt "\n", ##__VA_ARGS__);\
}while(0)

#define SLAM_FATAL(fmt, ...) do{\
    printf("[" __FILE__ ":" LOGLINE(__LINE__) "]  " fmt "\n", ##__VA_ARGS__);\
}while(0)
