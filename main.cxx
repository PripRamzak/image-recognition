#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/imgproc.hpp>

#include <iostream>

cv::RNG rng(12345);

using contour = std::vector<cv::Point>;

struct objects
{
    cv::Mat    img;
    contour    contour_;
    bool       have_similar;
    cv::Scalar color;
};

int main(int argc, char** argv)
{
    cv::CommandLineParser parser(
        argc, argv, "{@input | ../img/fruits4.jpg | input image}");
    cv::Mat src =
        cv::imread(cv::samples::findFile(parser.get<cv::String>("@input")),
                   cv::IMREAD_UNCHANGED);
    if (src.empty())
    {
        std::cout << "Could not open or find the image!\n" << std::endl;
        std::cout << "usage: " << argv[0] << " <Input image>" << std::endl;
        return -1;
    }

    // binarize image
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

    // smothing image
    cv::Mat filtred;
    cv::bilateralFilter(gray, filtred, 11, 17, 17);

    // find edges
    cv::Mat canny;
    cv::Canny(filtred, canny, 25., 50.);

    imshow("canny", canny);
    cv::waitKey(0);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
    cv::Mat closed;
    cv::morphologyEx(canny, closed, cv::MORPH_CLOSE, kernel);
    imshow("morph", closed);

    std::vector<contour> contours;
    cv::findContours(
        closed, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    std::vector<objects> templates;

    cv::Mat with_contours;
    src.copyTo(with_contours);

    for (int i = 0; i < contours.size(); i++)
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
                if (intersect_rect.area() / rect.area() > 0.75)
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

            templates.push_back({ gray(cv::Range(rect.y, rect.y + rect.height),
                                       cv::Range(rect.x, rect.x + rect.width)),
                                  contours[i],
                                  false,
                                  { 0., 0., 0. } });
            cv::rectangle(with_contours, rect, cv::Scalar(0., 255., 0.), 2);
        }
        else
        {
            contours.erase(contours.begin() + i);
            i--;
        }

    imshow("contours", with_contours);
    cv::waitKey(0);

    for (int i = 0; i < templates.size(); i++)
    {
        cv::Scalar color(rng.uniform(0., 255.),
                         rng.uniform(0., 255.),
                         rng.uniform(0., 255.));

        std::vector<int> indeces_of_similar = { i };

        for (int j = -1; j < 3; j++)
        {
            // rotate template
            cv::Mat res, templ_rotated;
            j != -1 ? cv::rotate(templates[i].img, templ_rotated, j)
                    : templates[i].img.copyTo(templ_rotated);
            imshow("template", templ_rotated);

            cv::matchTemplate(gray, templ_rotated, res, cv::TM_CCOEFF_NORMED);

            res.convertTo(res, CV_8U, 255.);
            imshow("match", res);

            double thresh =
                templ_rotated.rows * templ_rotated.cols < 10000 ? 0.9 : 0.675;
            cv::threshold(res, res, thresh * 255., 255., cv::THRESH_BINARY);
            imshow("result_thresh", res);

            // looking for similar objects
            std::vector<cv::Rect> similar;
            while (true)
            {
                double    maxval;
                cv::Point maxloc;
                minMaxLoc(res, NULL, &maxval, NULL, &maxloc);

                if (maxval > 0.)
                {
                    similar.emplace_back(
                        maxloc,
                        cv::Point(maxloc.x + templates[i].img.cols,
                                  maxloc.y + templates[i].img.rows));
                    floodFill(res, maxloc, 0);
                }
                else
                    break;
            }

            // looking for object that similar
            for (auto& sim : similar)
                for (int j = 0; j < templates.size(); j++)
                {
                    cv::Rect templ_rect =
                        cv::boundingRect(templates[j].contour_);
                    cv::Rect intersect_rect = templ_rect & sim;
                    double   rect_matching =
                        static_cast<double>(intersect_rect.area()) / sim.area();

                    if (rect_matching > 0.7)
                    {
                        if (j != i)
                        {
                            if (templates[j].have_similar)
                                color = templates[j].color;
                            indeces_of_similar.push_back(j);
                        }

                        break;
                    }
                }

            cv::waitKey(0);
        }

        if (indeces_of_similar.size() > 1)
            for (auto index : indeces_of_similar)
            {
                templates[index].have_similar = true;
                templates[index].color        = color;

                cv::RotatedRect min_rect =
                    cv::minAreaRect(templates[index].contour_);

                cv::Point2f points[4];
                min_rect.points(points);

                for (int k = 0; k < 4; k++)
                    line(src, points[k], points[(k + 1) % 4], color, 2);
            }

        imshow("after check template", src);
        cv::waitKey(0);
    }

    imshow("final", src);
    cv::waitKey(0);
}