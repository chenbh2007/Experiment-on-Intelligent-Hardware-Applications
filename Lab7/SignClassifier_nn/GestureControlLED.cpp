#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <opencv2/video/background_segm.hpp>
#include "toml.hpp"
#include "PCF8591.h"

#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#include <deque>
#include <filesystem>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <algorithm>

namespace fs = std::filesystem;

static volatile std::sig_atomic_t g_is_running = 1;

void handle_signal(int sig) {
    (void)sig;
    g_is_running = 0;
}

// 标签映射（严格对齐 SignClassifier.cpp）
// 0: Fist     -> 握拳 (全关 DAC=0)
// 1: LSign    -> L手势 (中等亮度 DAC=128)
// 2: VSign    -> V手势 (逐步调暗 DAC-=step)
// 3: Palm     -> 手掌 (全开 DAC=255)
// 4: Pointing -> 食指 (逐步调亮 DAC+=step)
static const char* GESTURE_NAMES[5] = {
    "Fist [OFF]",
    "LSign [MID 50%]",
    "VSign [DIM -]",
    "Palm [FULL ON]",
    "Pointing [BRIGHT +]"
};

// ============================================================================
// 1. PCF8591 I2C 硬件控制封装（支持自动降级为软件仿真模式）
// ============================================================================
class PCF8591Controller {
private:
    std::string bus_path_;
    uint8_t chip_addr_;
    int fd_;
    bool is_simulation_;
    int current_brightness_;

public:
    PCF8591Controller(const std::string& bus_path = "/dev/i2c-7", uint8_t chip_addr = 0x48)
        : bus_path_(bus_path), chip_addr_(chip_addr), fd_(-1), is_simulation_(false), current_brightness_(0) {
        fd_ = open(bus_path_.c_str(), O_RDWR);
        if (fd_ < 0) {
            std::cout << "[Hardware] Cannot open " << bus_path_
                      << ", switching to SIMULATION mode." << std::endl;
            is_simulation_ = true;
            return;
        }
        if (DA_output(fd_, chip_addr_, 0x00) < 0) {
            std::cout << "[Hardware] PCF8591 (0x" << std::hex << (int)chip_addr_ << std::dec
                      << ") did not respond on " << bus_path_
                      << ", switching to SIMULATION mode." << std::endl;
            close(fd_);
            fd_ = -1;
            is_simulation_ = true;
        } else {
            std::cout << "[Hardware] PCF8591 initialized on " << bus_path_
                      << " (addr 0x" << std::hex << (int)chip_addr_ << std::dec << ")." << std::endl;
        }
    }

    ~PCF8591Controller() {
        shutdown();
    }

    void set_brightness(int value) {
        int clamped = std::max(0, std::min(255, value));
        current_brightness_ = clamped;
        if (is_simulation_ || fd_ < 0) {
            return;
        }
        if (DA_output(fd_, chip_addr_, static_cast<uint8_t>(clamped)) < 0) {
            std::cerr << "\n[Hardware] I2C write failed, falling back to SIMULATION mode." << std::endl;
            is_simulation_ = true;
            close(fd_);
            fd_ = -1;
        }
    }

    int get_brightness() const {
        return current_brightness_;
    }

    bool is_simulation() const {
        return is_simulation_;
    }

    void shutdown() {
        if (fd_ >= 0 && !is_simulation_) {
            DA_output(fd_, chip_addr_, 0x00);
        }
        if (fd_ >= 0) {
            close(fd_);
            fd_ = -1;
            std::cout << "[Hardware] LED turned off (DAC=0) and I2C bus closed safely." << std::endl;
        }
    }
};

// ============================================================================
// 2. 手势滑动窗口防抖与调光冷却状态机
// ============================================================================
class GestureStateMachine {
private:
    size_t window_size_;
    float min_conf_;
    int cooldown_ms_;
    std::deque<int> history_;
    std::chrono::steady_clock::time_point last_step_time_;

public:
    GestureStateMachine(size_t window_size = 7, float min_conf = 0.60f, int cooldown_ms = 350)
        : window_size_(window_size), min_conf_(min_conf), cooldown_ms_(cooldown_ms) {
        last_step_time_ = std::chrono::steady_clock::now() - std::chrono::milliseconds(cooldown_ms_);
    }

    int update(int raw_label, float conf) {
        if (raw_label >= 0 && conf >= min_conf_) {
            history_.push_back(raw_label);
        } else {
            history_.push_back(-1);
        }
        if (history_.size() > window_size_) {
            history_.pop_front();
        }

        int counts[5] = {0, 0, 0, 0, 0};
        for (size_t i = 0; i < history_.size(); ++i) {
            int g = history_[i];
            if (g >= 0 && g < 5) {
                counts[g]++;
            }
        }

        int best_g = -1;
        int best_cnt = 0;
        for (int g = 0; g < 5; ++g) {
            if (counts[g] > best_cnt) {
                best_cnt = counts[g];
                best_g = g;
            }
        }

        int majority_threshold = static_cast<int>(window_size_ / 2) + 1;
        if (best_cnt >= majority_threshold) {
            return best_g;
        }
        return -1;
    }

    bool can_trigger(int stable_gesture) {
        if (stable_gesture == 2 || stable_gesture == 4) {
            std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
            long long elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_step_time_).count();
            if (elapsed >= cooldown_ms_) {
                last_step_time_ = now;
                return true;
            }
            return false;
        }
        return true;
    }
};

// ============================================================================
// 3. 动态目标跟踪器：结合背景差分法 (讲义 8.3/8.4.4) 与 CamShift (Lab6 track.cpp)
// ============================================================================
enum TrackMode {
    MODE_BG_SUBTRACT = 0, // 背景差分法动态跟踪 (BackgroundSubtractorKNN + 静态背景差分)
    MODE_CAMSHIFT    = 1  // Lab6 Section2 CamShift 直方图反向投影动态跟踪
};

struct TrackResult {
    bool target_found;                  // 是否跟踪到有效手部目标
    bool is_initializing_bg;            // 是否正在初始化采集背景
    cv::Rect track_box;                 // 裁剪送入 CNN 的跟踪框 (已对齐边界)
    cv::RotatedRect camshift_rot_box;   // CamShift 旋转矩形 (用于绘制黄色旋转框)
    std::vector<cv::Point> max_contour; // 背景差分最大运动轮廓
    cv::Mat fg_mask;                    // 前景掩码 / 反向投影概率图
    cv::Mat cnn_input_img;              // 最终送入 CNN 的裁剪图像
};

class DynamicHandTracker {
private:
    TrackMode mode_;
    bool mask_bg_to_gray_;

    // 背景差分法成员 (对应讲义 8.3 & 8.4.4)
    cv::Ptr<cv::BackgroundSubtractorKNN> knn_bs_;
    cv::Mat static_bg_float_;
    cv::Mat static_bg_gray_;
    int bg_warmup_frames_;
    int warmup_count_;
    cv::Rect smoothed_box_;
    bool has_smoothed_box_;

    // Lab6 CamShift 成员 (对应 Lab6/Section2/track.cpp)
    bool camshift_initialized_;
    cv::Rect camshift_window_;
    cv::Mat target_hist_;

public:
    DynamicHandTracker()
        : mode_(MODE_BG_SUBTRACT),
          mask_bg_to_gray_(false),
          bg_warmup_frames_(25),
          warmup_count_(0),
          has_smoothed_box_(false),
          camshift_initialized_(false) {
        knn_bs_ = cv::createBackgroundSubtractorKNN(100, 400.0, true);
    }

    void reset_background() {
        knn_bs_ = cv::createBackgroundSubtractorKNN(100, 400.0, true);
        static_bg_float_.release();
        static_bg_gray_.release();
        warmup_count_ = 0;
        has_smoothed_box_ = false;
        std::cout << "[Tracker] Resetting background model (please keep hand out of view for ~0.8s)..." << std::endl;
    }

    void toggle_mode(const cv::Mat& current_frame) {
        if (mode_ == MODE_BG_SUBTRACT) {
            mode_ = MODE_CAMSHIFT;
            camshift_initialized_ = false;
            camshift_window_ = cv::Rect(current_frame.cols / 3, current_frame.rows / 3,
                                        current_frame.cols / 3, current_frame.rows / 3);
            std::cout << "[Tracker] Switched to Lab6 CamShift mode. Press 'c' to lock target in red box." << std::endl;
        } else {
            mode_ = MODE_BG_SUBTRACT;
            std::cout << "[Tracker] Switched to Background Subtraction dynamic tracking mode." << std::endl;
        }
    }

    // 对应 Lab6/Section2/track.cpp 中按键锁定初始窗口并计算 HSV 直方图
    void init_camshift_from_box(const cv::Mat& bgr_frame, const cv::Rect& init_box) {
        cv::Rect safe_box = init_box & cv::Rect(0, 0, bgr_frame.cols, bgr_frame.rows);
        if (safe_box.width <= 10 || safe_box.height <= 10) {
            return;
        }
        cv::Mat target = bgr_frame(safe_box).clone();
        cv::cvtColor(target, target, cv::COLOR_BGR2HSV);

        const int histSize[] = {30, 32};
        float hranges[] = {0, 180};
        float sranges[] = {0, 256};
        const float* ranges[] = {hranges, sranges};
        const int channels[] = {0, 1};

        cv::calcHist(&target, 1, channels, cv::Mat(), target_hist_, 2, histSize, ranges);
        cv::normalize(target_hist_, target_hist_, 0, 255, cv::NORM_MINMAX);
        camshift_window_ = safe_box;
        camshift_initialized_ = true;
        mode_ = MODE_CAMSHIFT;
        std::cout << "[Tracker] CamShift histogram locked on window ("
                  << safe_box.x << "," << safe_box.y << ","
                  << safe_box.width << "x" << safe_box.height << ")." << std::endl;
    }

    void toggle_bg_gray_mask() {
        mask_bg_to_gray_ = !mask_bg_to_gray_;
    }

    bool is_bg_gray_mask_enabled() const {
        return mask_bg_to_gray_;
    }

    TrackMode get_mode() const {
        return mode_;
    }

    bool is_camshift_initialized() const {
        return camshift_initialized_;
    }

    // 将任意目标矩形扩展为带边距的正方形裁剪框（适配 120x120 手势训练集分布）
    static cv::Rect make_padded_square_box(const cv::Rect& raw_box, int frame_w, int frame_h) {
        int cx = raw_box.x + raw_box.width / 2;
        int cy = raw_box.y + raw_box.height / 2;
        int side = static_cast<int>(std::max(raw_box.width, raw_box.height) * 1.25);
        side = std::max(100, std::min({side, frame_w, frame_h}));

        int x1 = std::max(0, std::min(frame_w - side, cx - side / 2));
        int y1 = std::max(0, std::min(frame_h - side, cy - side / 2));
        return cv::Rect(x1, y1, side, side);
    }

    TrackResult process(const cv::Mat& frame) {
        TrackResult res;
        res.target_found = false;
        res.is_initializing_bg = false;
        int fw = frame.cols;
        int fh = frame.rows;
        cv::Rect center_box(fw / 3, fh / 3, fw / 3, fh / 3);
        res.track_box = center_box;

        // ====================================================================
        // 模式 A: 背景差分法动态跟踪 (讲义 8.3 & 8.4.4 BackgroundSubtractorKNN + 帧差)
        // ====================================================================
        if (mode_ == MODE_BG_SUBTRACT) {
            cv::Mat gray_blur;
            cv::cvtColor(frame, gray_blur, cv::COLOR_BGR2GRAY);
            cv::GaussianBlur(gray_blur, gray_blur, cv::Size(7, 7), 0);

            // 前 bg_warmup_frames_ 帧用于建立初始纯净背景模型（前 10 帧等待摄像头自动曝光稳定）
            if (warmup_count_ < bg_warmup_frames_) {
                res.is_initializing_bg = true;
                if (warmup_count_ == 10) {
                    knn_bs_ = cv::createBackgroundSubtractorKNN(100, 400.0, true);
                }
                if (warmup_count_ >= 10) {
                    cv::Mat tmp_mask;
                    knn_bs_->apply(frame, tmp_mask, 0.25); // 快速学习稳定曝光后的纯净背景
                    if (static_bg_float_.empty()) {
                        gray_blur.convertTo(static_bg_float_, CV_32FC1);
                    } else {
                        cv::accumulateWeighted(gray_blur, static_bg_float_, 0.25);
                    }
                    static_bg_float_.convertTo(static_bg_gray_, CV_8UC1);
                }
                warmup_count_++;
                res.fg_mask = cv::Mat::zeros(frame.size(), CV_8UC1);
                res.cnn_input_img = frame(center_box).clone();
                return res;
            }

            // 1. KNN 背景差分（参照讲义 8.4.4: apply -> threshold(244, 255) -> dilate -> findContours）
            // 注意：此处将 learningRate 设为 0.0，防止手部悬停做手势时被吸收进背景模型
            cv::Mat knn_mask, th_knn;
            knn_bs_->apply(frame, knn_mask, 0.0);
            cv::threshold(knn_mask, th_knn, 244, 255, cv::THRESH_BINARY);

            // 2. 结合静态背景帧差分 (参照讲义 8.3 图 8.5: |current - bg| > T)
            cv::Mat diff_gray, th_diff;
            cv::absdiff(gray_blur, static_bg_gray_, diff_gray);
            cv::threshold(diff_gray, th_diff, 30, 255, cv::THRESH_BINARY);

            // 融合两种差分前景掩码并进行形态学开运算去噪与膨胀连通 (讲义 8.4.4)
            cv::bitwise_and(th_knn, th_diff, res.fg_mask);
            cv::Mat kernel_open = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
            cv::Mat kernel_dilate = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
            cv::morphologyEx(res.fg_mask, res.fg_mask, cv::MORPH_OPEN, kernel_open, cv::Point(-1, -1), 1);
            cv::dilate(res.fg_mask, res.fg_mask, kernel_dilate, cv::Point(-1, -1), 2);

            // 3. 提取运动物体轮廓并筛选最大面积轮廓 (讲义 8.4.4: 使用 contourArea() 判断大小)
            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(res.fg_mask.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

            double max_area = 0.0;
            int max_idx = -1;
            for (size_t i = 0; i < contours.size(); ++i) {
                double area = cv::contourArea(contours[i]);
                if (area > max_area) {
                    max_area = area;
                    max_idx = static_cast<int>(i);
                }
            }

            // 面积门限：前景目标至少占画面的 1.5% 以上才确认为有效手部进入画面
            double min_valid_area = fw * fh * 0.015;
            if (max_idx >= 0 && max_area >= min_valid_area) {
                res.target_found = true;
                res.max_contour = contours[max_idx];
                cv::Rect raw_bbox = cv::boundingRect(res.max_contour);
                cv::Rect square_box = make_padded_square_box(raw_bbox, fw, fh);

                // 对跟踪框位置做指数平滑 (EMA)，使动态跟踪框移动平滑不抖动
                if (!has_smoothed_box_) {
                    smoothed_box_ = square_box;
                    has_smoothed_box_ = true;
                } else {
                    const double alpha = 0.45;
                    int nx = static_cast<int>(smoothed_box_.x * (1.0 - alpha) + square_box.x * alpha);
                    int ny = static_cast<int>(smoothed_box_.y * (1.0 - alpha) + square_box.y * alpha);
                    int nw = static_cast<int>(smoothed_box_.width * (1.0 - alpha) + square_box.width * alpha);
                    int nh = static_cast<int>(smoothed_box_.height * (1.0 - alpha) + square_box.height * alpha);
                    smoothed_box_ = cv::Rect(nx, ny, nw, nh) & cv::Rect(0, 0, fw, fh);
                }
                res.track_box = smoothed_box_;

                // 只保留最大轮廓的掩码供可选背景替换使用
                cv::Mat clean_fg = cv::Mat::zeros(frame.size(), CV_8UC1);
                cv::drawContours(clean_fg, contours, max_idx, cv::Scalar(255), cv::FILLED);
                cv::dilate(clean_fg, clean_fg, kernel_dilate, cv::Point(-1, -1), 2);
                res.fg_mask = clean_fg;
            } else {
                // 无手部目标时，缓慢自适应环境光照微小变化
                has_smoothed_box_ = false;
                cv::Mat adapt_mask;
                knn_bs_->apply(frame, adapt_mask, 0.02);
                cv::accumulateWeighted(gray_blur, static_bg_float_, 0.02);
                static_bg_float_.convertTo(static_bg_gray_, CV_8UC1);
            }

            if (mask_bg_to_gray_ && res.target_found) {
                cv::Mat gray_bg_canvas(res.track_box.size(), frame.type(), cv::Scalar(180, 180, 180));
                frame(res.track_box).copyTo(gray_bg_canvas, res.fg_mask(res.track_box));
                res.cnn_input_img = gray_bg_canvas;
            } else {
                res.cnn_input_img = frame(res.track_box).clone();
            }
            return res;
        }

        // ====================================================================
        // 模式 B: Lab6 Section2 CamShift 动态跟踪 (完全对齐 Lab6/Section2/track.cpp)
        // ====================================================================
        if (!camshift_initialized_) {
            camshift_window_ = center_box;
            res.track_box = center_box;
            res.fg_mask = cv::Mat::zeros(frame.size(), CV_8UC1);
            res.cnn_input_img = frame(center_box).clone();
            return res;
        }

        cv::Mat hsv, backproj;
        cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
        float hranges[] = {0, 180};
        float sranges[] = {0, 256};
        const float* ranges[] = {hranges, sranges};
        const int channels[] = {0, 1};

        cv::calcBackProject(&hsv, 1, channels, target_hist_, backproj, ranges);
        res.fg_mask = backproj;

        if (camshift_window_.width <= 0 || camshift_window_.height <= 0) {
            camshift_window_ = center_box;
        }
        camshift_window_ &= cv::Rect(0, 0, fw, fh);

        cv::RotatedRect rot_box = cv::CamShift(
            backproj, camshift_window_,
            cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 1000, 1.0));

        res.camshift_rot_box = rot_box;
        cv::Rect bound = rot_box.boundingRect() & cv::Rect(0, 0, fw, fh);
        if (bound.width >= 30 && bound.height >= 30) {
            res.target_found = true;
            camshift_window_ = bound;
            res.track_box = make_padded_square_box(bound, fw, fh);
        } else {
            res.target_found = false;
            res.track_box = center_box;
        }

        res.cnn_input_img = frame(res.track_box).clone();
        return res;
    }
};

// ============================================================================
// 4. 图像预处理与模型构建（严格对齐 SignClassifier.cpp）
// ============================================================================
torch::Tensor preprocess_frame(const cv::Mat& roi, int width, int height, torch::Device device) {
    cv::Mat img;
    cv::resize(roi, img, cv::Size(width, height));
    cv::cvtColor(img, img, cv::COLOR_BGR2HSV);

    std::vector<cv::Mat> img_channels;
    cv::split(img, img_channels);
    img_channels[0].convertTo(img_channels[0], CV_32F, 1.0 / 180.0);
    img_channels[1].convertTo(img_channels[1], CV_32F, 1.0 / 255.0);
    img_channels[2].convertTo(img_channels[2], CV_32F, 1.0 / 255.0);
    cv::merge(img_channels, img);

    torch::Tensor img_tensor = torch::from_blob(img.data, {1, height, width, 3}, torch::kFloat32).clone();
    img_tensor = img_tensor.permute({0, 3, 1, 2}).to(device);
    return img_tensor;
}

torch::nn::Sequential build_network(int w, int h, int conv_size, int conv_kernel_size,
                                    int pool_kernel_size, int linear_size, int linear_width) {
    torch::nn::Sequential net;
    int current_w = w;
    int current_h = h;
    int current_c = 3;

    net->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 3, conv_kernel_size)));
    net->push_back(torch::nn::ReLU());
    current_w += 1 - conv_kernel_size;
    current_h += 1 - conv_kernel_size;

    for (int i = 0; i < conv_size; i++) {
        net->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(current_c, 3, conv_kernel_size)));
        net->push_back(torch::nn::ReLU());
        current_w += 1 - conv_kernel_size;
        current_h += 1 - conv_kernel_size;
        net->push_back(torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions(pool_kernel_size)));
        current_w /= pool_kernel_size;
        current_h /= pool_kernel_size;
    }

    net->push_back(torch::nn::Flatten());
    net->push_back(torch::nn::Linear(current_c * current_w * current_h, linear_width));
    net->push_back(torch::nn::ReLU());

    for (int i = 0; i < linear_size; i++) {
        net->push_back(torch::nn::Linear(linear_width, linear_width));
        net->push_back(torch::nn::ReLU());
    }

    net->push_back(torch::nn::Linear(linear_width, 5));
    return net;
}

std::string resolve_existing_path(const std::string& candidate, const fs::path& config_dir) {
    if (fs::exists(candidate)) {
        return candidate;
    }
    fs::path p1 = fs::path("..") / candidate;
    if (fs::exists(p1)) {
        return p1.string();
    }
    fs::path p2 = config_dir / fs::path(candidate).filename();
    if (fs::exists(p2)) {
        return p2.string();
    }
    return candidate;
}

// ============================================================================
// 5. OpenCV HighGUI HUD 叠加绘制（展示动态跟踪框、前景掩码与 CNN 输入预览）
// ============================================================================
void draw_hud(cv::Mat& frame, const DynamicHandTracker& tracker,
              const TrackResult& track_res,
              int raw_label, float conf, int stable_label,
              int brightness, bool is_sim, double fps) {
    int h = frame.rows;
    int w = frame.cols;

    // 1. 绘制动态跟踪框与轮廓
    if (tracker.get_mode() == MODE_BG_SUBTRACT) {
        if (track_res.is_initializing_bg) {
            cv::rectangle(frame, track_res.track_box, cv::Scalar(0, 165, 255), 2);
            cv::putText(frame, "Learning Background... Keep Hand Out",
                        cv::Point(track_res.track_box.x, std::max(90, track_res.track_box.y - 10)),
                        cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 165, 255), 2, cv::LINE_AA);
        } else if (track_res.target_found) {
            // 绘制运动轮廓与动态跟踪框
            if (!track_res.max_contour.empty()) {
                std::vector<std::vector<cv::Point>> contours_vec(1, track_res.max_contour);
                cv::drawContours(frame, contours_vec, 0, cv::Scalar(0, 255, 128), 2, cv::LINE_AA);
            }
            cv::Scalar box_color = (stable_label >= 0) ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 215, 255);
            cv::rectangle(frame, track_res.track_box, box_color, 2);
            cv::putText(frame, "BG-Diff Tracked Hand",
                        cv::Point(track_res.track_box.x, std::max(88, track_res.track_box.y - 8)),
                        cv::FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2, cv::LINE_AA);
        }
    } else {
        // Lab6 CamShift 模式（与 Lab6/Section2/track.cpp 一致）
        if (!tracker.is_camshift_initialized()) {
            cv::rectangle(frame, track_res.track_box, cv::Scalar(0, 0, 255), 2);
            cv::putText(frame, "Put Hand Here & Press 'c' to Lock CamShift",
                        cv::Point(std::max(10, track_res.track_box.x - 40), std::max(88, track_res.track_box.y - 8)),
                        cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
        } else if (track_res.target_found) {
            cv::Point2f vertices[4];
            track_res.camshift_rot_box.points(vertices);
            for (int i = 0; i < 4; i++) {
                cv::line(frame, vertices[i], vertices[(i + 1) % 4], cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
            }
            cv::rectangle(frame, track_res.track_box, cv::Scalar(0, 255, 0), 2);
            cv::putText(frame, "CamShift Tracked ROI",
                        cv::Point(track_res.track_box.x, std::max(88, track_res.track_box.y - 8)),
                        cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
        }
    }

    // 2. 右上角并排绘制「前景差分掩码 (FG Mask)」与「CNN 输入预览 (CNN Input)」两个小窗 (各 88x66)
    if (w >= 300 && h >= 240) {
        int tw = 88;
        int th = 66;
        int ty = 80;

        if (!track_res.fg_mask.empty()) {
            int tx1 = w - tw * 2 - 24;
            cv::Mat fg_thumb, fg_bgr;
            cv::resize(track_res.fg_mask, fg_thumb, cv::Size(tw, th));
            cv::cvtColor(fg_thumb, fg_bgr, cv::COLOR_GRAY2BGR);
            fg_bgr.copyTo(frame(cv::Rect(tx1, ty, tw, th)));
            cv::rectangle(frame, cv::Rect(tx1, ty, tw, th), cv::Scalar(255, 200, 0), 1);
            cv::putText(frame, "FG Mask", cv::Point(tx1, ty + th + 14),
                        cv::FONT_HERSHEY_SIMPLEX, 0.42, cv::Scalar(255, 200, 0), 1, cv::LINE_AA);
        }

        if (!track_res.cnn_input_img.empty()) {
            int tx2 = w - tw - 12;
            cv::Mat cnn_thumb;
            cv::resize(track_res.cnn_input_img, cnn_thumb, cv::Size(tw, th));
            cnn_thumb.copyTo(frame(cv::Rect(tx2, ty, tw, th)));
            cv::rectangle(frame, cv::Rect(tx2, ty, tw, th), cv::Scalar(0, 255, 255), 1);
            cv::putText(frame, "CNN Input", cv::Point(tx2, ty + th + 14),
                        cv::FONT_HERSHEY_SIMPLEX, 0.42, cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
        }
    }

    // 3. 绘制顶部与底部半透明背景栏
    cv::Mat overlay = frame.clone();
    cv::rectangle(overlay, cv::Rect(0, 0, w, 72), cv::Scalar(20, 20, 20), -1);
    cv::rectangle(overlay, cv::Rect(0, h - 72, w, 72), cv::Scalar(20, 20, 20), -1);
    cv::addWeighted(overlay, 0.65, frame, 0.35, 0, frame);

    // 4. 顶部手势与置信度信息
    std::string status_str;
    cv::Scalar status_color;
    if (track_res.is_initializing_bg) {
        status_str = "Capturing Background Model...";
        status_color = cv::Scalar(0, 165, 255);
    } else if (!track_res.target_found) {
        status_str = (tracker.get_mode() == MODE_BG_SUBTRACT)
            ? "No Moving Hand (Press 'b' to Reset BG)"
            : "Press 'c' to Lock CamShift Target";
        status_color = cv::Scalar(185, 185, 185);
    } else if (stable_label >= 0 && stable_label < 5) {
        status_str = std::string("Stable: ") + GESTURE_NAMES[stable_label];
        status_color = cv::Scalar(0, 255, 120);
    } else if (raw_label >= 0 && raw_label < 5) {
        status_str = std::string("Detecting: ") + GESTURE_NAMES[raw_label] + " ...";
        status_color = cv::Scalar(0, 215, 255);
    } else {
        status_str = "Tracking Hand...";
        status_color = cv::Scalar(190, 190, 190);
    }
    cv::putText(frame, status_str, cv::Point(16, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.72, status_color, 2, cv::LINE_AA);

    std::ostringstream conf_ss;
    conf_ss << "Conf: " << std::fixed << std::setprecision(1)
            << (track_res.target_found ? conf * 100.0f : 0.0f)
            << "% | Mode(m): " << (tracker.get_mode() == MODE_BG_SUBTRACT ? "BG-Subtractor" : "Lab6-CamShift")
            << " | BGMask(f): " << (tracker.is_bg_gray_mask_enabled() ? "ON" : "OFF");
    cv::putText(frame, conf_ss.str(), cv::Point(16, 58),
                cv::FONT_HERSHEY_SIMPLEX, 0.48, cv::Scalar(220, 220, 220), 1, cv::LINE_AA);

    // 右上角硬件状态与 FPS
    std::string hw_str = is_sim ? "I2C: SIMULATION" : "I2C: /dev/i2c-7 [HW]";
    cv::Scalar hw_color = is_sim ? cv::Scalar(0, 165, 255) : cv::Scalar(0, 255, 0);
    cv::putText(frame, hw_str, cv::Point(std::max(10, w - 205), 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.52, hw_color, 1, cv::LINE_AA);

    std::ostringstream fps_ss;
    fps_ss << "FPS: " << std::fixed << std::setprecision(1) << fps;
    cv::putText(frame, fps_ss.str(), cv::Point(std::max(10, w - 205), 58),
                cv::FONT_HERSHEY_SIMPLEX, 0.52, cv::Scalar(220, 220, 220), 1, cv::LINE_AA);

    // 5. 底部 LED 亮度进度条
    double pct = static_cast<double>(brightness) / 255.0;
    int bar_x = 16;
    int bar_y = h - 34;
    int bar_w = w - 32;
    int bar_h = 20;

    cv::rectangle(frame, cv::Rect(bar_x, bar_y, bar_w, bar_h), cv::Scalar(55, 55, 55), -1);
    int fill_w = static_cast<int>(bar_w * pct);
    if (fill_w > 0) {
        cv::Scalar fill_color(0, static_cast<int>(160 + 95 * pct), 255);
        cv::rectangle(frame, cv::Rect(bar_x, bar_y, fill_w, bar_h), fill_color, -1);
    }
    cv::rectangle(frame, cv::Rect(bar_x, bar_y, bar_w, bar_h), cv::Scalar(130, 130, 130), 1);

    std::ostringstream led_ss;
    led_ss << "LED Brightness: " << std::fixed << std::setprecision(1) << (pct * 100.0)
           << "% (DAC: " << brightness << " / 255)";
    cv::putText(frame, led_ss.str(), cv::Point(bar_x, bar_y - 10),
                cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

    cv::putText(frame, "[b:ResetBG | m:BG/CamShift | c:LockCamShift | f:Mask | q:Quit]",
                cv::Point(std::max(10, w - 430), bar_y - 10),
                cv::FONT_HERSHEY_SIMPLEX, 0.42, cv::Scalar(180, 180, 180), 1, cv::LINE_AA);
}

// ============================================================================
// 6. 主函数
// ============================================================================
int main(int argc, char** argv) {
    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);

    std::string config_file = "c2ck3p4n1nw64/config.toml";
    std::string model_override = "";
    std::string cam_arg = "/dev/video0";
    std::string i2c_bus = "/dev/i2c-7";

    std::vector<std::string> pos_args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
            config_file = argv[++i];
        } else if (arg == "--model" && i + 1 < argc) {
            model_override = argv[++i];
        } else if ((arg == "--camera" || arg == "--video") && i + 1 < argc) {
            cam_arg = argv[++i];
        } else if (arg == "--i2c" && i + 1 < argc) {
            i2c_bus = argv[++i];
        } else {
            pos_args.push_back(arg);
        }
    }
    if (pos_args.size() >= 1) config_file = pos_args[0];
    if (pos_args.size() >= 2) cam_arg = pos_args[1];
    if (pos_args.size() >= 3) i2c_bus = pos_args[2];

    config_file = resolve_existing_path(config_file, fs::current_path());

    // 1. 解析 TOML 配置文件
    int w = 80;
    int h = 60;
    int conv_size = 2;
    int conv_kernel_size = 3;
    int pool_kernel_size = 4;
    int linear_size = 1;
    int linear_width = 64;
    std::string device_str = "cuda";
    std::string best_state_file = "c2ck3p4n1nw64/best.pth";
    std::string state_file = "c2ck3p4n1nw64/baseline.pth";
    fs::path config_dir = fs::path(config_file).parent_path();

    if (fs::exists(config_file)) {
        std::cout << "[Config] Loading config from: " << config_file << std::endl;
        toml::table config = toml::parse_file(config_file);
        w = config["control"]["width"].value_or(80);
        h = config["control"]["height"].value_or(60);
        device_str = config["control"]["device"].value_or("cuda");
        best_state_file = config["control"]["best_state_file"].value_or("c2ck3p4n1nw64/best.pth");
        state_file = config["control"]["state_file"].value_or("c2ck3p4n1nw64/baseline.pth");
        conv_size = config["nn"]["num_conv_layers"].value_or(2);
        conv_kernel_size = config["nn"]["conv_kernel_size"].value_or(3);
        pool_kernel_size = config["nn"]["pool_kernel_size"].value_or(4);
        linear_size = config["nn"]["num_linear_layers"].value_or(1);
        linear_width = config["nn"]["linear_width"].value_or(64);
    } else {
        std::cout << "[Config] Config file not found (" << config_file
                  << "), using default c2ck3p4n1nw64 parameters." << std::endl;
    }

    // 2. 选择推理设备
    torch::Device device(torch::kCPU);
    if (device_str == "cuda" && torch::cuda::is_available()) {
        device = torch::Device(torch::kCUDA);
        std::cout << "[Device] CUDA is available. Running inference on GPU." << std::endl;
    } else {
        std::cout << "[Device] Running inference on CPU." << std::endl;
    }

    // 3. 构建并加载神经网络模型
    torch::nn::Sequential net = build_network(
        w, h, conv_size, conv_kernel_size, pool_kernel_size, linear_size, linear_width);

    if (!model_override.empty()) {
        best_state_file = model_override;
    }
    std::string model_path = resolve_existing_path(best_state_file, config_dir);
    if (!fs::exists(model_path)) {
        model_path = resolve_existing_path(state_file, config_dir);
    }
    if (!fs::exists(model_path)) {
        std::cerr << "[Error] Model weight file not found: " << best_state_file << std::endl;
        return -1;
    }

    try {
        torch::load(net, model_path, device);
        net->to(device);
        net->eval();
        std::cout << "[Model] Successfully loaded weights from: " << model_path << std::endl;
    } catch (const c10::Error& e) {
        std::cerr << "[Error] Failed to load model weights: " << e.what() << std::endl;
        return -1;
    }

    // 4. 初始化 PCF8591 硬件控制器、状态机与背景差分跟踪器
    PCF8591Controller led(i2c_bus, 0x48);
    GestureStateMachine state_machine(7, 0.60f, 350);
    DynamicHandTracker tracker;
    const int brightness_step = 25;

    // 5. 打开摄像头或视频源（默认 /dev/video0）
    cv::VideoCapture cap;
    bool is_camera_index = !cam_arg.empty() && std::all_of(cam_arg.begin(), cam_arg.end(), ::isdigit);
    if (is_camera_index) {
        std::string dev_node = "/dev/video" + cam_arg;
        if (fs::exists(dev_node)) {
            cap.open(dev_node);
        } else {
            cap.open(std::stoi(cam_arg));
        }
    } else {
        cap.open(cam_arg);
    }

    if (!cap.isOpened()) {
        std::cerr << "[Error] Failed to open video source: " << cam_arg << std::endl;
        return -1;
    }

    // 6. 检查 X11 / HighGUI 显示环境
    const char* display_env = std::getenv("DISPLAY");
    bool has_gui = (display_env != nullptr && std::string(display_env).length() > 0);
    const std::string window_name = "Atlas 200I DK A2 - Gesture LED Control";
    if (has_gui) {
        std::cout << "[Display] DISPLAY=" << display_env
                  << " detected. Opening HighGUI window via X11..." << std::endl;
        cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
    } else {
        std::cout << "[Display] No DISPLAY environment variable detected. Running in terminal mode." << std::endl;
    }

    std::cout << "[Tracker] Initializing background model for first 15 frames (press 'b' anytime to re-capture background)." << std::endl;

    int fps_frames = 0;
    double current_fps = 0.0;
    std::chrono::steady_clock::time_point fps_start = std::chrono::steady_clock::now();

    torch::NoGradGuard no_grad;
    cv::Mat frame;

    while (g_is_running) {
        if (!cap.read(frame) || frame.empty()) {
            break;
        }

        // 1. 背景差分 / CamShift 动态目标跟踪
        TrackResult track_res = tracker.process(frame);

        int raw_label = -1;
        float raw_conf = 0.0f;

        // 2. 当跟踪到有效前景手部目标时，裁剪跟踪框送入 CNN 推理
        if (track_res.target_found && !track_res.cnn_input_img.empty()) {
            torch::Tensor input_tensor = preprocess_frame(track_res.cnn_input_img, w, h, device);
            torch::Tensor logits = net->forward(input_tensor);
            torch::Tensor probs = torch::softmax(logits, 1);

            std::tuple<torch::Tensor, torch::Tensor> max_res = probs.max(1);
            raw_conf = std::get<0>(max_res).item<float>();
            raw_label = static_cast<int>(std::get<1>(max_res).item<int64_t>());
        }

        // 3. 状态机多数表决防抖滤波
        int stable_gesture = state_machine.update(raw_label, raw_conf);

        // 4. 根据稳定手势控制 LED 亮度
        if (stable_gesture >= 0 && state_machine.can_trigger(stable_gesture)) {
            int cur_val = led.get_brightness();
            if (stable_gesture == 3) {         // Palm -> 全开 (255)
                led.set_brightness(255);
            } else if (stable_gesture == 0) {  // Fist -> 全关 (0)
                led.set_brightness(0);
            } else if (stable_gesture == 4) {  // Pointing -> 调亮 (+step)
                led.set_brightness(cur_val + brightness_step);
            } else if (stable_gesture == 2) {  // VSign -> 调暗 (-step)
                led.set_brightness(cur_val - brightness_step);
            } else if (stable_gesture == 1) {  // LSign -> 中等亮度 (128)
                led.set_brightness(128);
            }
        }

        // 5. 更新 FPS
        fps_frames++;
        std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
        double elapsed_sec = std::chrono::duration<double>(now - fps_start).count();
        if (elapsed_sec >= 0.5) {
            current_fps = fps_frames / elapsed_sec;
            fps_frames = 0;
            fps_start = now;
        }

        // 6. 渲染 HUD 与按键交互
        int cur_brightness = led.get_brightness();
        if (has_gui) {
            draw_hud(frame, tracker, track_res, raw_label, raw_conf,
                     stable_gesture, cur_brightness, led.is_simulation(), current_fps);
            cv::imshow(window_name, frame);
            int key = cv::waitKey(1) & 0xFF;
            if (key == 'q' || key == 27) {
                std::cout << "\n[Exit] Quit key pressed." << std::endl;
                break;
            } else if (key == 'b') {
                // 重新采集背景帧
                tracker.reset_background();
            } else if (key == 'm') {
                // 在「背景差分动态跟踪」与「Lab6 CamShift 跟踪」之间切换
                tracker.toggle_mode(frame);
            } else if (key == 'c') {
                // 锁定当前框（或中心框）初始化 Lab6 CamShift 直方图跟踪
                cv::Rect init_box = track_res.target_found
                    ? track_res.track_box
                    : cv::Rect(frame.cols / 3, frame.rows / 3, frame.cols / 3, frame.rows / 3);
                tracker.init_camshift_from_box(frame, init_box);
            } else if (key == 'f') {
                // 切换是否将非前景区域替换为浅灰背景
                tracker.toggle_bg_gray_mask();
            } else if (key == 'r') {
                led.set_brightness(0);
            }
        } else {
            const char* g_str = track_res.is_initializing_bg
                ? "Init Background..."
                : (!track_res.target_found
                    ? "No Moving Target"
                    : ((stable_gesture >= 0) ? GESTURE_NAMES[stable_gesture] : "Tracking..."));
            int bars = (cur_brightness * 20) / 255;
            std::string bar_str(bars, '#');
            bar_str.append(20 - bars, '-');
            std::cout << "\r" << (led.is_simulation() ? "[SIM] " : "[HW]  ")
                      << "State: " << std::left << std::setw(20) << g_str
                      << " Box:(" << std::setw(3) << track_res.track_box.x << ","
                      << std::setw(3) << track_res.track_box.y << ")"
                      << " Conf:" << std::fixed << std::setprecision(1) << std::setw(5) << (raw_conf * 100.0f) << "% | "
                      << "LED: [" << bar_str << "] " << std::right << std::setw(3) << cur_brightness << "/255"
                      << std::flush;
        }
    }

    if (!has_gui) {
        std::cout << std::endl;
    }
    cap.release();
    if (has_gui) {
        cv::destroyAllWindows();
    }
    led.shutdown();
    return 0;
}
