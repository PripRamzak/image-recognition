#pragma once

#include <opencv4/opencv2/core.hpp>

namespace image_recognition
{
using contour = std::vector<cv::Point>;

struct object
{
    cv::Mat    img;
    contour    contour_;
    cv::Rect   bounding_rect;
    cv::Scalar color        = { 0., 0., 0. };
    bool       have_similar = false;
};

void find_objects(cv::Mat&             gray,
                  std::vector<object>& objects,
                  double               low_threshold,
                  double               high_threshold,
                  bool                 show_intermidiate_result = false);

void multi_template_matching(cv::Mat&               gray,
                             cv::Mat&               templ,
                             std::vector<cv::Rect>& rects_of_similar,
                             double                 thresh_small_object,
                             double                 thresh_big_object,
                             bool show_intermieiate_result = false);

void find_similar(cv::Mat&             gray,
                  cv::Mat&             output,
                  std::vector<object>& objects,
                  double               small_object_similarity_limit,
                  double               big_object_similarity_limit,
                  bool                 show_intermediate_result = false);

void draw_contours(cv::Mat& src, std::vector<object>& objects);

} // namespace image_recognition