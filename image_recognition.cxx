#include "image_recognition.hxx"

#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgproc.hpp>

namespace image_recognition
{
static cv::RNG rng(12345);

static void draw_min_rect(cv::Mat&         output,
                          cv::RotatedRect& rect,
                          cv::Scalar&      color);

void find_objects(cv::Mat&             gray,
                  std::vector<object>& objects,
                  double               low_threshold,
                  double               high_threshold,
                  bool                 show_intermediate_result)
{
    // find edges
    cv::Mat canny;
    cv::Canny(gray, canny, low_threshold, high_threshold);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
    cv::Mat closed;
    cv::morphologyEx(canny, closed, cv::MORPH_CLOSE, kernel);

    std::vector<contour> contours;
    cv::findContours(
        closed, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    for (int i = 0; i < contours.size(); i++)
    {
        // check for very small contours (99% trash)
        if (cv::contourArea(contours[i]) > 100.)
        {
            cv::Rect rect = cv::boundingRect(contours[i]);

            bool part_of_object = false;
            for (int j = 0; j < contours.size(); j++)
            {
                if (j == i)
                    continue;

                // check for inner contours and delete them
                cv::Rect intersect_rect = cv::boundingRect(contours[j]) & rect;
                if (intersect_rect.area() / rect.area() > 0.8)
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
                                rect });
        }
        else
        {
            contours.erase(contours.begin() + i);
            i--;
        }
    }

    if (show_intermediate_result)
    {
        imshow("canny", canny);
        imshow("morph", closed);
        cv::waitKey(0);
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
        imshow("template", templ);
        imshow("match", result_match);
        imshow("result_thresh", result_thresh);
        cv::waitKey(0);
    }
}

void find_similar(cv::Mat&             gray,
                  cv::Mat&             output,
                  std::vector<object>& objects,
                  double               thresh_small_object,
                  double               thresh_big_object,
                  bool                 show_intermediate_result)
{
    for (int i = 0; i < objects.size(); i++)
    {
        cv::Scalar color(rng.uniform(0., 256.),
                         rng.uniform(0., 256.),
                         rng.uniform(0., 256.));

        std::vector<int> indeces_of_similar = { i };

        for (int j = -1; j < 3; j++)
        {
            // rotate template
            cv::Mat templ;
            j != -1 ? cv::rotate(objects[i].img, templ, j)
                    : objects[i].img.copyTo(templ);

            std::vector<cv::Rect> rects_of_similar;
            multi_template_matching(gray,
                                    templ,
                                    rects_of_similar,
                                    thresh_small_object,
                                    thresh_big_object,
                                    show_intermediate_result);

            // looking for object that similar
            for (auto& rect : rects_of_similar)
                for (int j = 0; j < objects.size(); j++)
                {
                    cv::Rect intersect_rect = objects[j].bounding_rect & rect;
                    double   rect_matching =
                        static_cast<double>(intersect_rect.area()) /
                        rect.area();

                    if (rect_matching > 0.7)
                    {
                        if (j != i)
                        {
                            if (objects[j].have_similar)
                                color = objects[j].color;
                            indeces_of_similar.push_back(j);
                        }

                        break;
                    }
                }
        }

        if (indeces_of_similar.size() > 1)
            for (auto index : indeces_of_similar)
            {
                objects[index].have_similar = true;
                objects[index].color        = color;

                cv::RotatedRect min_rect =
                    cv::minAreaRect(objects[index].contour_);
                draw_min_rect(output, min_rect, color);
            }

        if (show_intermediate_result)
        {
            imshow("after check template", output);
            cv::waitKey(0);
        }
    }
}

void draw_contours(cv::Mat& src, std::vector<object>& objects)
{
    for (auto& object_ : objects)
    {
        cv::Rect   rect = cv::boundingRect(object_.contour_);
        cv::Scalar color(rng.uniform(0., 256.),
                         rng.uniform(0., 256.),
                         rng.uniform(0., 256.));
        cv::rectangle(src, rect, color, 2);
    }
}

void draw_min_rect(cv::Mat& output, cv::RotatedRect& rect, cv::Scalar& color)
{
    cv::Point2f points[4];
    rect.points(points);

    for (int k = 0; k < 4; k++)
        line(output, points[k], points[(k + 1) % 4], color, 2);
}

} // namespace image_recognition