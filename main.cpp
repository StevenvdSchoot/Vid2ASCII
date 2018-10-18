#include <QCoreApplication>
#include <QDir>

#include <memory>
#include <thread>

#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>

#include <fcntl.h>
#include <unistd.h>
#include <ncurses.h>

static constexpr bool collect = false;
static constexpr bool train = false;
static constexpr bool visualize = false;

int main(int argc, char *argv[])
{
    cv::CascadeClassifier classifier{"/usr/share/OpenCV/haarcascades/haarcascade_frontalface_alt2.xml"};
    auto faceRecognizer = cv::face::LBPHFaceRecognizer::create();

    if constexpr(train) {
        auto categories = QDir{"faces"}.entryList();
        std::vector<cv::Mat> images;
        std::vector<int> labels;
        foreach(auto category, categories) {
            bool flag = false;
            int label = category.toInt(&flag);
            if(!flag)
                continue;
            auto filenames = QDir{QStringLiteral("faces/") + category}.entryList();
            foreach(auto filename, filenames) {
                auto img = cv::imread((QStringLiteral("faces/") + category + '/' + filename).toStdString());
                if(img.empty())
                    continue;
                std::vector<cv::Rect> faces;
                classifier.detectMultiScale(img, faces);
                if(faces.size() != 1)
                    continue;
                std::cout << "Using file: " << ((QStringLiteral("faces/") + category + '/' + filename).toStdString()) << " with label " << label << std::endl;
                cv::Mat gray_image;
                cv::cvtColor(img(faces[0]), gray_image, CV_BGR2GRAY);
                images.emplace_back(std::move(gray_image));
                labels.push_back(label);
            }
        }
        std::cout << "Start training using " << images.size() << " faces..." << std::endl;
        faceRecognizer->train(images, labels);
        faceRecognizer->write("data.yml");
    } else {
        faceRecognizer->read("data.yml");
    }

    std::string currentFolder;
    {
        char cCurrentPath[1024];
        getcwd(static_cast<char*>(cCurrentPath), sizeof(cCurrentPath));
        currentFolder += static_cast<char*>(cCurrentPath);
    }

    std::string cmd("jp2a --color --background=dark ");

    initscr();
    const bool useColors = has_colors();

    if(useColors) {
        start_color();
        init_pair(1, COLOR_YELLOW, COLOR_BLACK);
        init_pair(2, COLOR_GREEN, COLOR_BLACK);
    }

    const auto [rows, cols] = [](){
        int row, col;
        getmaxyx(stdscr, row, col);
        return std::make_tuple(row, col);
    }();

    cmd += "--height=";
    cmd += std::to_string(rows);
    cmd += " --width=";
    cmd += std::to_string(cols-1);
    cmd += " ";
    cmd += currentFolder + "/image.jpeg";
    const char* cmdPtr = cmd.c_str();

    cv::VideoCapture cap;
    cv::Mat img;

    const auto start = std::chrono::steady_clock::now() + std::chrono::milliseconds(5000);

    bool syncFrames = false;
    if(1 < argc) {
        std::string cmd("youtube-dl -o video \"");
        cmd += argv[1];
        cmd += '"';
        system(cmd.c_str());
        cap.open(currentFolder + "/video.mkv");
        syncFrames = true;
        pid_t pID = fork();
        if(pID == 0) {
            int fdout = open("/dev/null", O_WRONLY);
            dup2(fdout, STDOUT_FILENO);
            dup2(fdout, STDERR_FILENO);
            std::this_thread::sleep_until(start);
            execl("/usr/bin/ffplay", "ffplay", "-nodisp", "-vn", "./video.mkv", (char*)0);
        }
    } else {
        cap.open(0);
    }

    const std::chrono::milliseconds skipFrameDuration(static_cast<long>(1000. / cap.get(CV_CAP_PROP_FPS)));

    int faceCounter = 0;

    while(cap.isOpened()) {
        cap >> img;
        int globalClassification = -1;
        cv::Rect globalFaceRect;

        if(syncFrames) {
            const auto endpoint = start + std::chrono::milliseconds(static_cast<long>(cap.get(CV_CAP_PROP_POS_MSEC)));
            if(skipFrameDuration < std::chrono::steady_clock::now() - endpoint)
                continue;
            std::this_thread::sleep_until(endpoint);
        } else {
            {
                std::vector<cv::Rect> faces;
                classifier.detectMultiScale(img, faces);

                cv::Mat vis;
                if constexpr(visualize)
                    vis = img.clone();

                cv::Mat mask{img.size(), CV_32F, 0.};
                for(const auto& face : faces) {
                    cv::Mat greyFace;
                    cv::cvtColor(img(face), greyFace, CV_BGR2GRAY);
                    int classification;
                    double confidence;
                    faceRecognizer->predict(greyFace, classification, confidence);
                    if constexpr(visualize) {
                        cv::putText(vis, std::to_string(classification), face.tl(), cv::FONT_HERSHEY_PLAIN, 1, CV_RGB(0, 0, 255));
                        cv::putText(vis, std::to_string(confidence), face.tl() + cv::Point(50,0), cv::FONT_HERSHEY_PLAIN, 1, CV_RGB(0, 0, 255));
                    }

                    auto color = CV_RGB(255, 215, 0);
                    if(30. < confidence && (globalFaceRect.area() < face.area() || globalClassification <= -1)) {
                        globalFaceRect = face;
                        globalClassification = classification;
                        if(classification == 0)
                            color = CV_RGB(0, 255, 0);
                    }

                    {
                        cv::Mat maskModel, bgdModel, fgdModel;
                        maskModel.create(img.size(), CV_8UC1);
                        maskModel.setTo(cv::Scalar::all(cv::GC_BGD));
                        cv::grabCut(img, maskModel, globalFaceRect, bgdModel, fgdModel, 1, cv::GC_INIT_WITH_RECT);
                        cv::Mat m = (maskModel & 1) * 255;
                        cv::add(mask, m, mask, cv::noArray(), CV_32F);
                    }

                    if constexpr(visualize)
                        cv::rectangle(vis, face, color, 1);

                    if constexpr(collect) {
                        cv::imwrite(std::string{"faces/0/face "} + std::to_string(faceCounter) + ".jpeg", img(face));
                        faceCounter++;
                    }
                }

                if constexpr(visualize) {
                    if(0 <= globalClassification) {
                        mask = mask.mul(.5);
                        mask += 255./2.;
                        cv::GaussianBlur(mask, mask, cv::Size(49,49), 0.);
                        cv::Mat m;
                        mask.convertTo(m, CV_8U);
                        cv::cvtColor(m, m, cv::COLOR_GRAY2BGR);
                        vis = img.mul(m, 1./255.);
                    }
                    cv::imshow("Capture", vis);
                    cv::waitKey(1);
                }
            }
        }

        cv::imwrite("image.jpeg", img);
        std::shared_ptr<FILE> pipe(popen(cmdPtr, "r"), pclose);
        if (!pipe) throw std::runtime_error("popen() failed!");
        std::string result;
        while (!feof(pipe.get())) {
            std::array<char, 1024*1024> buffer;
            if (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr)
                result += buffer.data();
        }

        mvprintw(0, 0, result.c_str());
        if(!syncFrames && 0 <= globalClassification) {
            if(useColors) {
                if(globalClassification == 0)
                    attron(COLOR_PAIR(2));
                else
                    attron(COLOR_PAIR(1));
            }
            auto classificationString = std::to_string(globalClassification);
            mvprintw(rows - 1, cols - static_cast<int>(classificationString.size()), classificationString.c_str());
            if(useColors) {
                if(globalClassification == 0)
                    attroff(COLOR_PAIR(2));
                else
                    attroff(COLOR_PAIR(1));
            }
        }

        refresh();
    }

    return 0;
}
