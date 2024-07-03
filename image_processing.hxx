#pragma once

#include <opencv4/opencv2/core.hpp>

namespace image_processing
{
using contour = std::vector<cv::Point>;

struct object
{
    cv::Mat    img;
    contour    contour_;
    cv::Rect   bounding_rect;
    cv::Scalar color;
    bool       have_similar = false;
};

double rect_matching(cv::Rect rect1, cv::Rect rect2);

void find_edges(cv::Mat& gray,
                cv::Mat& edges,
                double   low_threshe,
                double   high_thresh,
                bool     show_intermediate_result = false);

void find_objects(cv::Mat&              gray,
                  std::vector<contour>& contours,
                  std::vector<object>&  objects);

void multi_template_matching(cv::Mat&               gray,
                             cv::Mat&               templ,
                             std::vector<cv::Rect>& rects_of_similar,
                             double                 thresh_small_object,
                             double                 thresh_big_object,
                             bool show_intermediate_result = false);

void draw_min_rect(cv::Mat& output, object& object);

} // namespace image_processing