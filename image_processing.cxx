#include "image_processing.hxx"

#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgproc.hpp>

#include <algorithm>

namespace image_processing
{
static cv::RNG rng(12345);

double rect_matching(cv::Rect rect1, cv::Rect rect2)
{
    cv::Rect intersect_rect = rect1 & rect2;
    return static_cast<double>(intersect_rect.area()) / rect2.area();
}

void find_edges(cv::Mat& gray,
                cv::Mat& edges,
                double   low_thresh,
                double   high_thresh,
                bool     show_intermediate_result)
{
    cv::Mat canny;
    cv::Canny(gray, canny, low_thresh, high_thresh);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
    cv::morphologyEx(canny, edges, cv::MORPH_CLOSE, kernel);

    if (show_intermediate_result)
    {
        imshow("canny", canny);
        cv::waitKey(0);
    }
}

void find_objects(cv::Mat&              gray,
                  std::vector<contour>& contours,
                  std::vector<object>&  objects)
{
    // check for very small contours (99% trash)
    contours.erase(std::remove_if(contours.begin(),
                                  contours.end(),
                                  [](auto& contour)
                                  { return cv::contourArea(contour) <= 100.; }),
                   contours.end());

    for (int i = 0; i < contours.size(); i++)
    {
        cv::Rect rect = cv::boundingRect(contours[i]);

        bool part_of_object = false;
        for (int j = 0; j < contours.size(); j++)
        {
            if (j == i)
                continue;

            // check for inner contours and delete them
            if (rect_matching(cv::boundingRect(contours[j]), rect) > 0.75)
            {
                part_of_object = true;
                break;
            }
        }

        if (part_of_object)
        {
            contours.erase(contours.begin() + i);
            i--;
            continue;
        }

        objects.push_back({ gray(cv::Range(rect.y, rect.y + rect.height),
                                 cv::Range(rect.x, rect.x + rect.width)),
                            contours[i],
                            rect,
                            { rng.uniform(0., 256.),
                              rng.uniform(0., 256.),
                              rng.uniform(0., 256.) } });
    }
}

void multi_template_matching(cv::Mat&               gray,
                             cv::Mat&               templ,
                             std::vector<cv::Rect>& rects_of_similar,
                             double                 thresh_small_object,
                             double                 thresh_big_object,
                             bool                   show_intermediate_result)
{
    cv::Mat result_match;
    cv::matchTemplate(gray, templ, result_match, cv::TM_CCOEFF_NORMED);

    cv::Mat result_thresh;
    double  thresh = templ.rows * templ.cols < 10000 ? thresh_small_object
                                                     : thresh_big_object;
    cv::threshold(result_match, result_thresh, thresh, 1., cv::THRESH_BINARY);

    // looking for similar objects
    while (true)
    {
        double    maxval;
        cv::Point maxloc;
        minMaxLoc(result_thresh, NULL, &maxval, NULL, &maxloc);

        if (maxval > 0.)
        {
            rects_of_similar.emplace_back(
                maxloc,
                cv::Point(maxloc.x + templ.cols, maxloc.y + templ.rows));
            floodFill(result_thresh, maxloc, 0);
        }
        else
            break;
    }

    if (show_intermediate_result)
    {
        imshow("rotated", templ);
        imshow("match", result_match);
        imshow("result_thresh", result_thresh);
        cv::waitKey(0);
    }
}

void draw_min_rect(cv::Mat& output, object& object)
{
    cv::RotatedRect min_rect = cv::minAreaRect(object.contour_);

    cv::Point2f points[4];
    min_rect.points(points);

    for (int k = 0; k < 4; k++)
        line(output, points[k], points[(k + 1) % 4], object.color, 2);
}

} // namespace image_processing