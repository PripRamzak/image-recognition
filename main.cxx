#include "image_processing.hxx"

#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgproc.hpp>

#include <iostream>

int main(int argc, char** argv)
{
    cv::CommandLineParser parser(
        argc, argv, "{@input | ../img/fruits5.jpg | input image}");
    cv::Mat src =
        cv::imread(cv::samples::findFile(parser.get<cv::String>("@input")),
                   cv::IMREAD_UNCHANGED);
    if (src.empty())
    {
        std::cout << "Could not open or find the image!\n" << std::endl;
        std::cout << "usage: " << argv[0] << " <Input image>" << std::endl;
        return EXIT_FAILURE;
    }

    // binarize image
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    imshow("gray", gray);

    // smothing image
    cv::Mat filtred;
    cv::bilateralFilter(gray, filtred, 11, 17., 17.);

    cv::Mat edges;
    image_processing::find_edges(filtred, edges, 25., 50., true);

    std::vector<image_processing::contour> contours;
    cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    std::vector<image_processing::object> objects;
    image_processing::find_objects(gray, contours, objects);

    cv::Mat all_objects;
    src.copyTo(all_objects);
    for (auto& object : objects)
        image_processing::draw_min_rect(all_objects, object);

    cv::imshow("all_objects", all_objects);

    cv::Mat result;
    src.copyTo(result);

    for (int i = 0; i < objects.size(); i++)
    {
        std::vector<int> indeces_of_similar = { i };
        imshow("template", objects[i].img);

        for (int j = -1; j < 3; j++)
        {
            // rotate template
            cv::Mat templ;
            j != -1 ? cv::rotate(objects[i].img, templ, j)
                    : objects[i].img.copyTo(templ);

            std::vector<cv::Rect> rects_of_similar;
            image_processing::multi_template_matching(
                gray, templ, rects_of_similar, 0.9, 0.675);

            // looking for object that similar
            for (auto& rect : rects_of_similar)
                for (int k = 0; k < objects.size(); k++)
                    if (image_processing::rect_matching(
                            objects[k].bounding_rect, rect) > 0.7)
                    {
                        if (k != i)
                        {
                            if (objects[k].have_similar)
                                objects[i].color = objects[k].color;
                            indeces_of_similar.push_back(k);
                        }

                        break;
                    }
        }

        if (indeces_of_similar.size() > 1)
            for (auto index : indeces_of_similar)
            {
                objects[index].have_similar = true;
                objects[index].color        = objects[i].color;

                draw_min_rect(result, objects[index]);
            }

        imshow("after check template", result);
        cv::waitKey(0);
    }

    imshow("result", result);
    cv::waitKey(0);
}