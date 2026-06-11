#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>


#include <nav_msgs/msg/odometry.hpp>
#include <pluginlib/class_list_macros.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <std_msgs/msg/float64.hpp>
#include <std_msgs/msg/u_int8.hpp>
#include <vehicle_controller_interfaces/command_state_base.hpp>
#include <vehicle_controller_interfaces/feedback_state_base.hpp>
#include <vehicle_controller_interfaces/upper_body_trajectory_handler.hpp>

#include <cd110r_controller/cd110r_command_state.hpp>
#include <cd110r_controller/cd110r_feedback_state.hpp>
#include <cd110r_controller/cd110r_joy_mapping.hpp>

namespace cd110r_plugins {

class Cd110UpperBodyTrajectoryHandler final
    : public vehicle_controller_interfaces::UpperBodyTrajectoryHandler {
   public:
    // ============================================================
    // @brief Initialize the upper body trajectory handler.
    // ============================================================
    Cd110UpperBodyTrajectoryHandler() = default;

    enum class SwingControlState : uint8_t {
        kIdle = 0U,
        kFreeSwing = 1U,
        kSwing180 = 2U,
        kCenterCheck = 3U,
        kRecoverNegative = 4U,
        kRecoverPositive = 5U,
        kConstSwing = 6U,
    };

    struct PidControlResult {
        double lever{0.0};
        double p_control{0.0};
        double i_control{0.0};
        double d_control{0.0};
    };

    // ============================================================
    // @brief Initialize parameters, subscriptions, and debug publishers.
    // ============================================================
    void Initialize(
        rclcpp::Node& node,
        const std::shared_ptr<vehicle_controller_interfaces::CommandStateBase>&
            command_state,
        const std::shared_ptr<vehicle_controller_interfaces::FeedbackStateBase>&
            feedback_state) override {
        node_ = &node;
        command_state_ =
            std::dynamic_pointer_cast<cd110r_controller::CD110RCommandState>(
                command_state);
        feedback_state_ =
            std::dynamic_pointer_cast<cd110r_controller::CD110RFeedbackState>(
                feedback_state);

        node_->declare_parameter<double>("upper_body_trajectory_timeout_sec",
                                         90.0);
        node_->declare_parameter<double>("upper_body_switch_wait_timeout_sec",
                                         3.0);
        node_->declare_parameter<double>("upper_body_switch_on_threshold", 0.5);
        node_->declare_parameter<double>("upper_body_switch_stable_sec", 1.0);
        node_->declare_parameter<double>(
            "upper_body_reverse_toggle_threshold_deg", 90.0);
        node_->declare_parameter<double>(
            "upper_body_rotate_medium_axis_command", 0.8);
        node_->declare_parameter<double>("upper_body_rotate_slow_axis_command",
                                         0.3);
        node_->declare_parameter<double>("upper_body_slow_down_threshold_deg",
                                         6.0);
        node_->declare_parameter<double>("upper_body_center_hold_window_deg",
                                         1.0);
        node_->declare_parameter<double>("upper_body_center_align_timeout_sec",
                                         12.0);
        node_->declare_parameter<std::string>(
            "upper_body_odom_topic",
            "vehicle_state_estimator/current_odom");
        node_->declare_parameter<std::string>("upper_body_imu_topic",
                                              "mtlt335_can_parser/imu_129");

        node_->declare_parameter<double>("upper_body_pid_kp", 0.032);
        node_->declare_parameter<double>("upper_body_pid_ki", 0.005);
        node_->declare_parameter<double>("upper_body_pid_kd", 0.035);
        node_->declare_parameter<double>("upper_body_pid_threshold_deg", 40.0);
        node_->declare_parameter<double>("upper_body_pid_threshold_imu", 0.05);
        node_->declare_parameter<double>("upper_body_pid_dec_target_angle_deg",
                                         45.0);
        node_->declare_parameter<double>("upper_body_pid_max_lever_abs", 1.0);
        node_->declare_parameter<double>("upper_body_pid_bias", 0.4);
        node_->declare_parameter<double>("upper_body_pid_min_lever_abs", 0.1);
        node_->declare_parameter<double>("upper_body_goal_tolerance_deg", 1.5);
        node_->declare_parameter<double>("upper_body_angle_diff_threshold_deg",
                                         45.0);
        node_->declare_parameter<int>("upper_body_velocity_median_window", 5);
        node_->declare_parameter<int>("upper_body_euler_median_window", 10);
        node_->declare_parameter<int>("upper_body_mean_window", 5);
        node_->declare_parameter<int>("upper_body_control_period_ms", 50);

        upper_body_trajectory_timeout_sec_ =
            node_->get_parameter("upper_body_trajectory_timeout_sec")
                .as_double();
        upper_body_switch_wait_timeout_sec_ =
            node_->get_parameter("upper_body_switch_wait_timeout_sec")
                .as_double();
        upper_body_switch_on_threshold_ =
            node_->get_parameter("upper_body_switch_on_threshold").as_double();
        upper_body_switch_stable_sec_ =
            node_->get_parameter("upper_body_switch_stable_sec").as_double();
        upper_body_reverse_toggle_threshold_deg_ =
            node_->get_parameter("upper_body_reverse_toggle_threshold_deg")
                .as_double();
        upper_body_center_align_timeout_sec_ =
            node_->get_parameter("upper_body_center_align_timeout_sec")
                .as_double();
        upper_body_odom_topic_ =
            node_->get_parameter("upper_body_odom_topic").as_string();
        upper_body_imu_topic_ =
            node_->get_parameter("upper_body_imu_topic").as_string();

        upper_body_pid_kp_ =
            node_->get_parameter("upper_body_pid_kp").as_double();
        upper_body_pid_ki_ =
            node_->get_parameter("upper_body_pid_ki").as_double();
        upper_body_pid_kd_ =
            node_->get_parameter("upper_body_pid_kd").as_double();
        upper_body_pid_threshold_deg_ =
            node_->get_parameter("upper_body_pid_threshold_deg").as_double();
        upper_body_pid_threshold_imu_ =
            node_->get_parameter("upper_body_pid_threshold_imu").as_double();
        upper_body_pid_dec_target_angle_deg_ =
            node_->get_parameter("upper_body_pid_dec_target_angle_deg")
                .as_double();
        upper_body_pid_max_lever_abs_ =
            node_->get_parameter("upper_body_pid_max_lever_abs").as_double();
        upper_body_pid_bias_ =
            node_->get_parameter("upper_body_pid_bias").as_double();
        upper_body_pid_min_lever_abs_ =
            node_->get_parameter("upper_body_pid_min_lever_abs").as_double();
        upper_body_goal_tolerance_deg_ =
            node_->get_parameter("upper_body_goal_tolerance_deg").as_double();
        upper_body_angle_diff_threshold_deg_ =
            node_->get_parameter("upper_body_angle_diff_threshold_deg")
                .as_double();
        const auto velocity_window_param =
            node_->get_parameter("upper_body_velocity_median_window").as_int();
        const auto euler_window_param =
            node_->get_parameter("upper_body_euler_median_window").as_int();
        const auto mean_window_param =
            node_->get_parameter("upper_body_mean_window").as_int();
        const auto control_period_param =
            node_->get_parameter("upper_body_control_period_ms").as_int();
        upper_body_velocity_median_window_ = static_cast<std::size_t>(
            std::max<int64_t>(1, velocity_window_param));
        upper_body_euler_median_window_ = static_cast<std::size_t>(
            std::max<int64_t>(1, euler_window_param));
        upper_body_mean_window_ = static_cast<std::size_t>(
            std::max<int64_t>(1, mean_window_param));
        upper_body_control_period_ms_ =
            static_cast<int>(std::max<int64_t>(1, control_period_param));

        if (upper_body_trajectory_timeout_sec_ <= 0.0) {
            upper_body_trajectory_timeout_sec_ = 90.0;
        }
        if (upper_body_switch_wait_timeout_sec_ <= 0.0) {
            upper_body_switch_wait_timeout_sec_ = 3.0;
        }
        if (upper_body_switch_on_threshold_ <= 0.0) {
            upper_body_switch_on_threshold_ = 0.5;
        }
        if (upper_body_switch_stable_sec_ < 0.0) {
            upper_body_switch_stable_sec_ = 1.0;
        }
        if (upper_body_reverse_toggle_threshold_deg_ < 0.0) {
            upper_body_reverse_toggle_threshold_deg_ = 90.0;
        }
        if (upper_body_center_align_timeout_sec_ <= 0.0) {
            upper_body_center_align_timeout_sec_ = 12.0;
        }
        if (upper_body_pid_kp_ < 0.0) {
            upper_body_pid_kp_ = 0.032;
        }
        if (upper_body_pid_ki_ < 0.0) {
            upper_body_pid_ki_ = 0.005;
        }
        if (upper_body_pid_kd_ < 0.0) {
            upper_body_pid_kd_ = 0.035;
        }
        if (upper_body_pid_threshold_deg_ <= 0.0) {
            upper_body_pid_threshold_deg_ = 40.0;
        }
        if (upper_body_pid_threshold_imu_ <= 0.0) {
            upper_body_pid_threshold_imu_ = 0.05;
        }
        if (upper_body_pid_dec_target_angle_deg_ <= 0.0) {
            upper_body_pid_dec_target_angle_deg_ = 45.0;
        }
        if (upper_body_pid_max_lever_abs_ <= 0.0) {
            upper_body_pid_max_lever_abs_ = 1.0;
        }
        if (upper_body_pid_bias_ < 0.0) {
            upper_body_pid_bias_ = 0.4;
        }
        if (upper_body_pid_min_lever_abs_ <= 0.0 ||
            upper_body_pid_min_lever_abs_ > upper_body_pid_max_lever_abs_) {
            upper_body_pid_min_lever_abs_ = 0.1;
        }
        if (upper_body_goal_tolerance_deg_ < 0.0) {
            upper_body_goal_tolerance_deg_ = 1.5;
        }
        if (upper_body_angle_diff_threshold_deg_ <= 0.0) {
            upper_body_angle_diff_threshold_deg_ = 45.0;
        }

        odom_sub_ = node_->create_subscription<nav_msgs::msg::Odometry>(
            upper_body_odom_topic_, rclcpp::QoS(100),
            [this](const nav_msgs::msg::Odometry::SharedPtr msg) {
                const double w = msg->pose.pose.orientation.w;
                const double x = msg->pose.pose.orientation.x;
                const double y = msg->pose.pose.orientation.y;
                const double z = msg->pose.pose.orientation.z;
                double yaw_deg =
                    -std::atan2(2.0 * (w * z + x * y),
                                1.0 - 2.0 * (y * y + z * z)) *
                    180.0 / 3.14159265358979323846;
                if (yaw_deg < 0.0) {
                    yaw_deg += 360.0;
                }
                std::lock_guard<std::mutex> lock(yaw_angle_mutex_);
                latest_yaw_angle_deg_ = yaw_deg;
                has_yaw_angle_ = true;
            });
        imu_sub_ = node_->create_subscription<sensor_msgs::msg::Imu>(
            upper_body_imu_topic_, rclcpp::QoS(100),
            [this](const sensor_msgs::msg::Imu::SharedPtr msg) {
                std::lock_guard<std::mutex> lock(imu_state_mutex_);
                latest_swing_velocity_rad_s_ = msg->angular_velocity.z;
                latest_angular_velocity_y_ = msg->angular_velocity.y;
                latest_linear_acceleration_x_ = msg->linear_acceleration.x;
                latest_linear_acceleration_y_ = msg->linear_acceleration.y;
                latest_linear_acceleration_z_ = msg->linear_acceleration.z;
                has_imu_linear_acceleration_ = true;
            });

        swing_angle_diff_pub_ =
            node_->create_publisher<std_msgs::msg::Float64>(
                "SwingAngleDiffFromState", 10);
        state_swing_pub_ = node_->create_publisher<std_msgs::msg::UInt8>(
            "state_swing", 10);
        finish_sw_pub_ = node_->create_publisher<std_msgs::msg::UInt8>(
            "finish_sw", 10);
        center_pass_sw_pub_ =
            node_->create_publisher<std_msgs::msg::Float64>("center_pass_sw",
                                                            10);
        target_swing_angle_pub_ =
            node_->create_publisher<std_msgs::msg::Float64>(
                "target_swing_angle", 10);
        swing_angle_diff_azimuth_pub_ =
            node_->create_publisher<std_msgs::msg::Float64>(
                "swing_angle_diff_azimuth", 10);
        yaw_angle_degrees_pub_ =
            node_->create_publisher<std_msgs::msg::Float64>(
                "yaw_angle_degrees", 10);
        linear_acceleration_magnitude_pub_ =
            node_->create_publisher<std_msgs::msg::Float64>(
                "linear_acceleration_magnitude", 10);
        linear_acceleration_x_mean_pub_ =
            node_->create_publisher<std_msgs::msg::Float64>(
                "linear_acceleration_x_mean", 10);
        linear_acceleration_y_mean_pub_ =
            node_->create_publisher<std_msgs::msg::Float64>(
                "linear_acceleration_y_mean", 10);
        linear_acceleration_z_mean_pub_ =
            node_->create_publisher<std_msgs::msg::Float64>(
                "linear_acceleration_z_mean", 10);
        angular_velocity_y_mean_pub_ =
            node_->create_publisher<std_msgs::msg::Float64>(
                "angular_velocity_y_mean", 10);
        linear_acceleration_magnitude_mean_pub_ =
            node_->create_publisher<std_msgs::msg::Float64>(
                "linear_acceleration_magnitude_mean", 10);
        p_control_pub_ = node_->create_publisher<std_msgs::msg::Float64>(
            "p_control", 10);
        i_control_pub_ = node_->create_publisher<std_msgs::msg::Float64>(
            "i_control", 10);
        d_control_pub_ = node_->create_publisher<std_msgs::msg::Float64>(
            "d_control", 10);
        swing_velocity_median_pub_ =
            node_->create_publisher<std_msgs::msg::Float64>(
                "swing_velocity_median", 10);
        euler_x_median_pub_ = node_->create_publisher<std_msgs::msg::Float64>(
            "euler_x_median", 10);
        euler_y_median_pub_ = node_->create_publisher<std_msgs::msg::Float64>(
            "euler_y_median", 10);
        e_integration_pub_ = node_->create_publisher<std_msgs::msg::Float64>(
            "e_integration", 10);
        current_state_swing_pub_ =
            node_->create_publisher<std_msgs::msg::UInt8>(
                "current_state_swing", 10);
    }

    // ============================================================
    // @brief Validate the received trajectory goal.
    // ============================================================
    bool IsGoalValid(const std::shared_ptr<const FollowJointTrajectory::Goal>&
                         goal) const override {
        if (!goal) {
            return false;
        }
        if (goal->trajectory.points.empty()) {
            return false;
        }
        const auto& point = goal->trajectory.points.back();
        return !point.positions.empty();
    }

    // ============================================================
    // @brief Stop the upper body command and reset debug values.
    // ============================================================
    void Cancel() override {
        command_state_->SetAxisValue(cd110r_controller::axis::kUpperBody, 0.0F);
        PublishDebugData(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0U, 0U, 0.0, 0U,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0);
    }

    // ============================================================
    // @brief Return whether a goal is currently being processed.
    // ============================================================
    bool IsBusy() const override { return goal_active_.load(); }

    // ============================================================
    // @brief Process an upper body goal with state transitions and PID control.
    // ============================================================
    void HandleGoal(const std::shared_ptr<rclcpp_action::ServerGoalHandle<
                        FollowJointTrajectory>>& goal_handle) override {
        // Mark goal processing as active and restore the flag on exit.
        goal_active_.store(true);
        auto goal_guard = std::shared_ptr<void>(
            nullptr, [this](void*) { goal_active_.store(false); });
        auto result = std::make_shared<FollowJointTrajectory::Result>();
        result->error_code = FollowJointTrajectory::Result::INVALID_GOAL;
        result->error_string = "Invalid goal";

        // Reject invalid goals with missing trajectory points or positions.
        const auto goal = goal_handle->get_goal();
        if (!IsGoalValid(goal)) {
            goal_handle->abort(result);
            return;
        }

        // Use the final trajectory point as the upper body target angle.
        const auto& point = goal->trajectory.points.back();
        constexpr double kPi = 3.14159265358979323846;
        constexpr double kRadToDeg = 180.0 / kPi;
        constexpr double kDegToRad = kPi / 180.0;
        const double target_rad = point.positions.front();
        const double target_deg = target_rad * kRadToDeg;

        // Only allow drive-compatible targets: 0 deg or 180 deg.
        if (!IsDriveCompatibleTargetDeg(target_deg)) {
            result->error_code = FollowJointTrajectory::Result::INVALID_GOAL;
            result->error_string =
                "Upper body target must be drive-compatible (0 or 180 deg)";
            goal_handle->abort(result);
            return;
        }
        const double move_duration_sec = std::max(
            0.1, static_cast<double>(point.time_from_start.sec) +
                     static_cast<double>(point.time_from_start.nanosec) * 1e-9);
        const double timeout_sec = std::max(upper_body_trajectory_timeout_sec_,
                                            move_duration_sec + 1.0);
        const auto switch_wait_timeout =
            std::chrono::duration<double>(upper_body_switch_wait_timeout_sec_);
        const auto switch_stable_window =
            std::chrono::duration<double>(upper_body_switch_stable_sec_);
        const auto control_period =
            std::chrono::milliseconds(upper_body_control_period_ms_);
        const bool reverse_toggle_requested =
            std::abs(target_deg) >= upper_body_reverse_toggle_threshold_deg_;

        // Wait until the initial switch states and upper body angle are ready.
        const auto switch_wait_start_tp = std::chrono::steady_clock::now();
        bool has_center_sw = false;
        bool has_reverse_sw = false;
        bool has_upper_body_angle = false;
        double center_sw_value = 0.0;
        double reverse_sw_value = 0.0;
        double upper_body_angle_deg = 0.0;
        while (rclcpp::ok()) {
            if (goal_handle->is_canceling()) {
                command_state_->SetAxisValue(
                    cd110r_controller::axis::kUpperBody, 0.0F);
                PublishDebugData(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0U, 0U, 0.0,
                                 0U, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                 0.0, 0.0, 0.0);
                result->error_code =
                    FollowJointTrajectory::Result::INVALID_GOAL;
                result->error_string = "Canceled";
                goal_handle->canceled(result);
                return;
            }

            {
                const auto snapshot = feedback_state_->GetSnapshot();
                has_center_sw = snapshot.has_center_sw;
                has_reverse_sw = snapshot.has_reverse_sw;
                center_sw_value = snapshot.center_sw;
                reverse_sw_value = snapshot.reverse_sw;
                has_upper_body_angle = snapshot.has_upper_body_angle;
                if (has_upper_body_angle) {
                    upper_body_angle_deg =
                        NormalizeAngleDeg(snapshot.upper_body_angle_deg);
                }
            }

            auto feedback = std::make_shared<FollowJointTrajectory::Feedback>();
            feedback->joint_names = goal->trajectory.joint_names;
            feedback->error.positions = {std::abs(target_rad)};
            goal_handle->publish_feedback(feedback);

            if (has_center_sw && has_reverse_sw && has_upper_body_angle) {
                break;
            }
            if ((std::chrono::steady_clock::now() - switch_wait_start_tp) >=
                switch_wait_timeout) {
                command_state_->SetAxisValue(
                    cd110r_controller::axis::kUpperBody, 0.0F);
                PublishDebugData(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0U, 0U, 0.0,
                                 0U, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                 0.0, 0.0, 0.0);
                result->error_code =
                    FollowJointTrajectory::Result::GOAL_TOLERANCE_VIOLATED;
                result->error_string =
                    "Timed out waiting "
                    "CD_CenterSw/CD_ReverseSw/upper_body_angle";
                goal_handle->abort(result);
                return;
            }
            std::this_thread::sleep_for(control_period);
        }

        bool start_center_on =
            has_center_sw && center_sw_value >= upper_body_switch_on_threshold_;
        bool center_seen_in_stable_window = start_center_on;
        if (switch_stable_window.count() > 0.0) {
            const auto stable_check_start_tp = std::chrono::steady_clock::now();
            while (rclcpp::ok()) {
                if (goal_handle->is_canceling()) {
                    command_state_->SetAxisValue(
                        cd110r_controller::axis::kUpperBody, 0.0F);
                    PublishDebugData(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0U, 0U,
                                     0.0, 0U, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                     0.0, 0.0, 0.0, 0.0, 0.0);
                    result->error_code =
                        FollowJointTrajectory::Result::INVALID_GOAL;
                    result->error_string = "Canceled";
                    goal_handle->canceled(result);
                    return;
                }

                {
                    const auto snapshot = feedback_state_->GetSnapshot();
                    has_center_sw = snapshot.has_center_sw;
                    has_reverse_sw = snapshot.has_reverse_sw;
                    center_sw_value = snapshot.center_sw;
                    reverse_sw_value = snapshot.reverse_sw;
                    has_upper_body_angle = snapshot.has_upper_body_angle;
                    if (has_upper_body_angle) {
                        upper_body_angle_deg =
                            NormalizeAngleDeg(snapshot.upper_body_angle_deg);
                    }
                }

                const bool center_on_now =
                    has_center_sw &&
                    center_sw_value >= upper_body_switch_on_threshold_;
                if (center_on_now) {
                    center_seen_in_stable_window = true;
                }
                if ((std::chrono::steady_clock::now() - stable_check_start_tp) >=
                    switch_stable_window) {
                    break;
                }
                std::this_thread::sleep_for(control_period);
            }
        }
        start_center_on = center_seen_in_stable_window;

        // Determine the final center angle from the current reverse state.
        const bool use_const_swing_recovery =
            !start_center_on && reverse_toggle_requested;
        const bool initial_reverse_state =
            reverse_sw_value >= upper_body_switch_on_threshold_;
        const bool target_reverse_state = reverse_toggle_requested
                                              ? !initial_reverse_state
                                              : initial_reverse_state;
        const double target_center_angle_deg = reverse_toggle_requested
                                                   ? (target_reverse_state
                                                          ? 180.0
                                                          : 0.0)
                                                   : NormalizeAngleDeg(target_deg);
        const bool target_is_drive_180_or_0 =
            std::abs(target_center_angle_deg) < 1e-6 ||
            std::abs(target_center_angle_deg - 180.0) < 1e-6;

        // Prepare buffers for median and mean statistics to reduce noise.
        std::vector<double> velocity_samples;
        velocity_samples.reserve(upper_body_velocity_median_window_);
        velocity_samples.push_back(0.0);
        auto push_velocity_sample = [&](double velocity_rad_s) {
            if (velocity_samples.size() >= upper_body_velocity_median_window_) {
                velocity_samples.erase(velocity_samples.begin());
            }
            velocity_samples.push_back(velocity_rad_s);
        };
        auto get_velocity_median = [&]() {
            if (velocity_samples.empty()) {
                return 0.0;
            }
            std::vector<double> sorted_samples = velocity_samples;
            std::sort(sorted_samples.begin(), sorted_samples.end());
            const std::size_t middle = sorted_samples.size() / 2U;
            if ((sorted_samples.size() % 2U) == 0U) {
                return (sorted_samples[middle - 1U] + sorted_samples[middle]) /
                       2.0;
            }
            return sorted_samples[middle];
        };
        std::vector<double> euler_x_samples;
        std::vector<double> euler_y_samples;
        std::vector<double> linear_acceleration_x_samples;
        std::vector<double> linear_acceleration_y_samples;
        std::vector<double> linear_acceleration_z_samples;
        std::vector<double> angular_velocity_y_samples;
        std::vector<double> linear_acceleration_magnitude_samples;
        euler_x_samples.reserve(upper_body_euler_median_window_);
        euler_y_samples.reserve(upper_body_euler_median_window_);
        linear_acceleration_x_samples.reserve(upper_body_mean_window_);
        linear_acceleration_y_samples.reserve(upper_body_mean_window_);
        linear_acceleration_z_samples.reserve(upper_body_mean_window_);
        angular_velocity_y_samples.reserve(upper_body_mean_window_);
        linear_acceleration_magnitude_samples.reserve(upper_body_mean_window_);
        euler_x_samples.push_back(0.0);
        euler_y_samples.push_back(0.0);
        linear_acceleration_x_samples.push_back(0.0);
        linear_acceleration_y_samples.push_back(0.0);
        linear_acceleration_z_samples.push_back(0.0);
        angular_velocity_y_samples.push_back(0.0);
        linear_acceleration_magnitude_samples.push_back(0.0);
        auto push_euler_sample = [&](std::vector<double>& samples,
                                     double value) {
            if (samples.size() >= upper_body_euler_median_window_) {
                samples.erase(samples.begin());
            }
            samples.push_back(value);
        };
        auto push_mean_sample = [&](std::vector<double>& samples,
                                    double value) {
            if (samples.size() >= upper_body_mean_window_) {
                samples.erase(samples.begin());
            }
            samples.push_back(value);
        };
        auto get_sample_median = [&](const std::vector<double>& samples) {
            if (samples.empty()) {
                return 0.0;
            }
            std::vector<double> sorted_samples = samples;
            std::sort(sorted_samples.begin(), sorted_samples.end());
            const std::size_t middle = sorted_samples.size() / 2U;
            if ((sorted_samples.size() % 2U) == 0U) {
                return (sorted_samples[middle - 1U] + sorted_samples[middle]) /
                       2.0;
            }
            return sorted_samples[middle];
        };
        auto get_sample_mean = [&](const std::vector<double>& samples) {
            if (samples.empty()) {
                return 0.0;
            }
            double sum = 0.0;
            for (const double value : samples) {
                sum += value;
            }
            return sum / static_cast<double>(samples.size());
        };

        // Initialize control state and the latest diagnostic values.
        const auto start_tp = std::chrono::steady_clock::now();
        auto slow_swing_start_tp = start_tp;
        auto center_check_start_tp = start_tp;
        double yaw_angle_deg = upper_body_angle_deg;
        bool has_yaw_angle = false;
        ReadYawAngle(has_yaw_angle, yaw_angle_deg);
        const double start_direction_deg = has_yaw_angle ? yaw_angle_deg
                                                         : upper_body_angle_deg;
        uint8_t state_swing = use_const_swing_recovery
                      ? 6U
                      : (target_is_drive_180_or_0 ? 2U : 1U);
        uint8_t current_state_swing = 0U;
        double e_integration = 0.0;
        double start_angle_diff = ComputeDriveAngleDiffDeg(
            target_center_angle_deg, upper_body_angle_deg,
            upper_body_angle_diff_threshold_deg_);
        bool center_pass_sw = false;
        double swing_lever = 0.0;
        bool latest_has_center_sw = has_center_sw;
        bool latest_has_reverse_sw = has_reverse_sw;
        bool latest_has_upper_body_angle = has_upper_body_angle;
        double latest_center_sw = center_sw_value;
        double latest_reverse_sw = reverse_sw_value;
        double latest_angle_deg = upper_body_angle_deg;
        double latest_swing_angle_diff_deg = start_angle_diff;
        double latest_velocity_median_rad_s = 0.0;
        double latest_p_control = 0.0;
        double latest_i_control = 0.0;
        double latest_d_control = 0.0;
        double latest_swing_angle_diff_azimuth_deg = 0.0;
        double latest_yaw_angle_deg = start_direction_deg;
        double latest_euler_x_median_deg = 0.0;
        double latest_euler_y_median_deg = 0.0;
        double latest_linear_acceleration_magnitude = 0.0;
        double latest_linear_acceleration_x_mean = 0.0;
        double latest_linear_acceleration_y_mean = 0.0;
        double latest_linear_acceleration_z_mean = 0.0;
        double latest_angular_velocity_y_mean = 0.0;
        double latest_linear_acceleration_magnitude_mean = 0.0;
        double latest_target_swing_angle_deg = target_center_angle_deg;
        while (rclcpp::ok()) {
            // On cancel, stop output and exit while preserving latest diagnostics.
            if (goal_handle->is_canceling()) {
                command_state_->SetAxisValue(
                    cd110r_controller::axis::kUpperBody, 0.0F);
                PublishDebugData(
                    0.0, 0.0, 0.0, 0.0, 0.0, e_integration, 0U, 0U, 0.0,
                    0U, 0.0, latest_yaw_angle_deg,
                    latest_euler_x_median_deg, latest_euler_y_median_deg,
                    latest_linear_acceleration_magnitude,
                    latest_linear_acceleration_x_mean,
                    latest_linear_acceleration_y_mean,
                    latest_linear_acceleration_z_mean,
                    latest_angular_velocity_y_mean,
                    latest_linear_acceleration_magnitude_mean,
                    latest_target_swing_angle_deg);
                result->error_code =
                    FollowJointTrajectory::Result::INVALID_GOAL;
                result->error_string = "Canceled";
                goal_handle->canceled(result);
                return;
            }

            const auto now_tp = std::chrono::steady_clock::now();
            const double elapsed_sec =
                std::chrono::duration<double>(now_tp - start_tp).count();

            {
                const auto snapshot = feedback_state_->GetSnapshot();
                has_center_sw = snapshot.has_center_sw;
                has_reverse_sw = snapshot.has_reverse_sw;
                center_sw_value = snapshot.center_sw;
                reverse_sw_value = snapshot.reverse_sw;
                has_upper_body_angle = snapshot.has_upper_body_angle;
                if (has_upper_body_angle) {
                    upper_body_angle_deg =
                        NormalizeAngleDeg(snapshot.upper_body_angle_deg);
                }
            }

            latest_has_center_sw = has_center_sw;
            latest_has_reverse_sw = has_reverse_sw;
            latest_has_upper_body_angle = has_upper_body_angle;
            latest_center_sw = center_sw_value;
            latest_reverse_sw = reverse_sw_value;
            latest_angle_deg = upper_body_angle_deg;
            ReadYawAngle(has_yaw_angle, yaw_angle_deg);
            latest_yaw_angle_deg = has_yaw_angle ? yaw_angle_deg
                                                 : latest_yaw_angle_deg;

            // Read IMU acceleration and angular velocity, then update statistics.
            bool has_imu = false;
            double linear_accel_x = 0.0;
            double linear_accel_y = 0.0;
            double linear_accel_z = 0.0;
            ReadImuLinearAcceleration(has_imu, linear_accel_x, linear_accel_y,
                                      linear_accel_z);
            if (has_imu) {
                const double linear_acceleration_magnitude = std::sqrt(
                    linear_accel_x * linear_accel_x +
                    linear_accel_y * linear_accel_y +
                    linear_accel_z * linear_accel_z);
                double euler_x_deg = 0.0;
                double euler_y_deg = 0.0;
                ComputeEulerAnglesDeg(linear_accel_x, linear_accel_y,
                                      linear_accel_z, euler_x_deg,
                                      euler_y_deg);
                push_euler_sample(euler_x_samples, euler_x_deg);
                push_euler_sample(euler_y_samples, euler_y_deg);
                double angular_velocity_y = 0.0;
                ReadImuAngularVelocityY(angular_velocity_y);
                push_mean_sample(linear_acceleration_x_samples,
                                 linear_accel_x);
                push_mean_sample(linear_acceleration_y_samples,
                                 linear_accel_y);
                push_mean_sample(linear_acceleration_z_samples,
                                 linear_accel_z);
                push_mean_sample(angular_velocity_y_samples,
                                 angular_velocity_y);
                push_mean_sample(linear_acceleration_magnitude_samples,
                                 linear_acceleration_magnitude);
                latest_euler_x_median_deg = get_sample_median(euler_x_samples);
                latest_euler_y_median_deg = get_sample_median(euler_y_samples);
                latest_linear_acceleration_magnitude =
                    linear_acceleration_magnitude;
                latest_linear_acceleration_x_mean =
                    get_sample_mean(linear_acceleration_x_samples);
                latest_linear_acceleration_y_mean =
                    get_sample_mean(linear_acceleration_y_samples);
                latest_linear_acceleration_z_mean =
                    get_sample_mean(linear_acceleration_z_samples);
                latest_angular_velocity_y_mean =
                    get_sample_mean(angular_velocity_y_samples);
                latest_linear_acceleration_magnitude_mean =
                    get_sample_mean(linear_acceleration_magnitude_samples);
            }

            bool has_swing_velocity = false;
            double swing_velocity_rad_s = 0.0;
            ReadSwingVelocity(has_swing_velocity, swing_velocity_rad_s);
            if (has_swing_velocity) {
                push_velocity_sample(swing_velocity_rad_s);
            }

            // Compute the angle error and median swing angular velocity.
            const bool center_on =
                has_center_sw &&
                center_sw_value >= upper_body_switch_on_threshold_;
            const uint8_t previous_state_swing = current_state_swing;
            const double swing_angle_diff_deg = ComputeDriveAngleDiffDeg(
                target_center_angle_deg, upper_body_angle_deg,
                upper_body_angle_diff_threshold_deg_);
            const double swing_velocity_median_rad_s = get_velocity_median();
            latest_swing_angle_diff_azimuth_deg =
                ComputeSwingAngleDiffAzimuthDeg(start_direction_deg,
                                                latest_yaw_angle_deg, 180.0);
            latest_target_swing_angle_deg = target_center_angle_deg;

            if (previous_state_swing != 3U) {
                center_check_start_tp = now_tp;
            }

            // Evaluate state transitions and determine the next state_swing.
            uint8_t next_state_swing = state_swing;
            if (previous_state_swing == 0U) {
                next_state_swing = state_swing;
            } else if (previous_state_swing == 1U &&
                swing_angle_diff_deg >= -1.5 && swing_angle_diff_deg <= 1.5) {
                next_state_swing = 0U;
            } else if (previous_state_swing == 2U) {
                if (swing_velocity_median_rad_s > 0.02 ||
                    swing_angle_diff_deg > 40.0) {
                    slow_swing_start_tp = now_tp;
                }
                const bool stop_180_by_azimuth =
                    swing_angle_diff_deg < 90.0 &&
                    latest_swing_angle_diff_azimuth_deg < -1.0;
                const bool stop_180_by_center =
                    swing_angle_diff_deg < 90.0 && center_on;
                const bool stop_180_by_timer =
                    (now_tp - slow_swing_start_tp) >=
                    std::chrono::duration<double>(5.0);
                if (stop_180_by_center) {
                    center_pass_sw = true;
                }
                if (stop_180_by_azimuth || stop_180_by_center ||
                    stop_180_by_timer) {
                    next_state_swing = 3U;
                }
            } else if (previous_state_swing == 6U) {
                if (swing_velocity_median_rad_s > 0.02) {
                    slow_swing_start_tp = now_tp;
                }
                const bool stop_const_by_center = center_on;
                const bool stop_const_by_timer =
                    (now_tp - slow_swing_start_tp) >=
                    std::chrono::duration<double>(5.0);
                if (stop_const_by_center) {
                    center_pass_sw = true;
                }
                if (stop_const_by_center || stop_const_by_timer) {
                    next_state_swing = 3U;
                }
            } else if (previous_state_swing == 3U) {
                if ((now_tp - center_check_start_tp) >=
                    std::chrono::duration<double>(2.0)) {
                    if (center_on) {
                        next_state_swing = 0U;
                    } else if (center_pass_sw ||
                               latest_swing_angle_diff_azimuth_deg < 0.0) {
                        next_state_swing = 5U;
                    } else {
                        next_state_swing = 4U;
                    }
                }
            } else {
                if (center_on) {
                    next_state_swing = 3U;
                } else if (!(swing_angle_diff_deg < 15.0)) {
                    next_state_swing = 4U;
                } else if (!(swing_angle_diff_deg > -15.0)) {
                    next_state_swing = 5U;
                }
            }

            state_swing = next_state_swing;

            // Keep the integrator active only during swing control.
            if (!(state_swing == 1U || state_swing == 2U || state_swing == 6U)) {
                e_integration = 0.0;
            }
            if (previous_state_swing == 0U &&
                (state_swing == 1U || state_swing == 2U || state_swing == 6U)) {
                start_angle_diff = swing_angle_diff_deg;
            }

            // Select either PID control or recovery logic based on the state.
            PidControlResult control_result;
            if (state_swing == 1U || state_swing == 2U) {
                control_result = ComputeDriveLikePidControl(
                    swing_angle_diff_deg, swing_velocity_median_rad_s,
                    e_integration, start_angle_diff);
                swing_lever = control_result.lever;
            } else if (state_swing == 6U) {
                swing_lever = ComputeConstSwingLever();
                control_result.lever = swing_lever;
            } else if (state_swing == 4U || state_swing == 5U) {
                swing_lever = ComputeRecoverLever(
                    state_swing, previous_state_swing,
                    swing_velocity_median_rad_s, swing_lever);
                control_result.lever = swing_lever;
            } else {
                swing_lever = 0.0;
                control_result = {};
            }

            if (state_swing < 1U) {
                swing_lever = 0.0;
                control_result.lever = 0.0;
            }

            command_state_->SetAxisValue(cd110r_controller::axis::kUpperBody,
                                         static_cast<float>(swing_lever));

            // Store the latest control result and publish debug topics.
            latest_swing_angle_diff_deg = swing_angle_diff_deg;
            latest_velocity_median_rad_s = swing_velocity_median_rad_s;
            latest_p_control = control_result.p_control;
            latest_i_control = control_result.i_control;
            latest_d_control = control_result.d_control;
            PublishDebugData(
                swing_angle_diff_deg, control_result.p_control,
                control_result.i_control, control_result.d_control,
                swing_velocity_median_rad_s, e_integration,
                state_swing, current_state_swing,
                center_pass_sw ? 1.0 : 0.0,
                state_swing == 0U ? 1U : 10U,
                latest_swing_angle_diff_azimuth_deg, latest_yaw_angle_deg,
                latest_euler_x_median_deg, latest_euler_y_median_deg,
                latest_linear_acceleration_magnitude,
                latest_linear_acceleration_x_mean,
                latest_linear_acceleration_y_mean,
                latest_linear_acceleration_z_mean,
                latest_angular_velocity_y_mean,
                latest_linear_acceleration_magnitude_mean,
                latest_target_swing_angle_deg);

            auto feedback = std::make_shared<FollowJointTrajectory::Feedback>();
            feedback->joint_names = goal->trajectory.joint_names;
            feedback->error.positions = {std::abs(swing_angle_diff_deg) *
                                         kDegToRad};
            goal_handle->publish_feedback(feedback);

            current_state_swing = state_swing;

            // State 0 means the target has been reached successfully.
            if (state_swing == 0U) {
                command_state_->SetAxisValue(
                    cd110r_controller::axis::kUpperBody, 0.0F);
                PublishDebugData(
                    swing_angle_diff_deg, 0.0, 0.0, 0.0,
                    swing_velocity_median_rad_s, e_integration,
                    0U, current_state_swing, center_pass_sw ? 1.0 : 0.0,
                    1U, latest_swing_angle_diff_azimuth_deg,
                    latest_yaw_angle_deg, latest_euler_x_median_deg,
                    latest_euler_y_median_deg,
                    latest_linear_acceleration_magnitude,
                    latest_linear_acceleration_x_mean,
                    latest_linear_acceleration_y_mean,
                    latest_linear_acceleration_z_mean,
                    latest_angular_velocity_y_mean,
                    latest_linear_acceleration_magnitude_mean,
                    latest_target_swing_angle_deg);
                result->error_code =
                    FollowJointTrajectory::Result::SUCCESSFUL;
                result->error_string = "";
                goal_handle->succeed(result);
                return;
            }

            // Abort with the latest diagnostics when the timeout is exceeded.
            if (elapsed_sec >= timeout_sec) {
                command_state_->SetAxisValue(
                    cd110r_controller::axis::kUpperBody, 0.0F);
                PublishDebugData(latest_swing_angle_diff_deg, latest_p_control,
                                 latest_i_control, latest_d_control,
                                 latest_velocity_median_rad_s, e_integration,
                                 state_swing, current_state_swing,
                                 center_pass_sw ? 1.0 : 0.0,
                                 10U,
                                 latest_swing_angle_diff_azimuth_deg,
                                 latest_yaw_angle_deg,
                                 latest_euler_x_median_deg,
                                 latest_euler_y_median_deg,
                                 latest_linear_acceleration_magnitude,
                                 latest_linear_acceleration_x_mean,
                                 latest_linear_acceleration_y_mean,
                                 latest_linear_acceleration_z_mean,
                                 latest_angular_velocity_y_mean,
                                 latest_linear_acceleration_magnitude_mean,
                                 latest_target_swing_angle_deg);
                result->error_code =
                    FollowJointTrajectory::Result::GOAL_TOLERANCE_VIOLATED;
                char error_msg[256];
                std::snprintf(
                    error_msg, sizeof(error_msg),
                    "Upper body PID rotation timed out "
                    "(has_center_sw=%s, center_sw=%.1f, has_reverse_sw=%s, "
                    "reverse_sw=%.1f, has_upper_body_angle=%s, "
                    "upper_body_angle_deg=%.1f, swing_angle_diff=%.2f, "
                    "swing_angle_diff_azimuth=%.2f, state=%u)",
                    latest_has_center_sw ? "true" : "false", latest_center_sw,
                    latest_has_reverse_sw ? "true" : "false",
                    latest_reverse_sw,
                    latest_has_upper_body_angle ? "true" : "false",
                    latest_angle_deg, latest_swing_angle_diff_deg,
                    latest_swing_angle_diff_azimuth_deg,
                    static_cast<unsigned>(current_state_swing));
                result->error_string = error_msg;
                goal_handle->abort(result);
                return;
            }

            std::this_thread::sleep_for(control_period);
        }

    // Stop the axis command and abort if ROS is shutting down.
    command_state_->SetAxisValue(cd110r_controller::axis::kUpperBody, 0.0F);
        PublishDebugData(0.0, 0.0, 0.0, 0.0, 0.0, e_integration, 0U, 0U,
                         0.0, 10U, 0.0, latest_yaw_angle_deg,
                         latest_euler_x_median_deg,
                         latest_euler_y_median_deg,
                         latest_linear_acceleration_magnitude,
                         latest_linear_acceleration_x_mean,
                         latest_linear_acceleration_y_mean,
                         latest_linear_acceleration_z_mean,
                         latest_angular_velocity_y_mean,
                         latest_linear_acceleration_magnitude_mean,
                         latest_target_swing_angle_deg);
        result->error_code = FollowJointTrajectory::Result::INVALID_GOAL;
        result->error_string = "ROS shutdown";
        goal_handle->abort(result);
    }

   private:
    // ============================================================
    // @brief Normalize an angle to the range $[0, 360)$ degrees.
    // ============================================================
    static double NormalizeAngleDeg(double angle_deg) {
        while (angle_deg >= 360.0) {
            angle_deg -= 360.0;
        }
        while (angle_deg < 0.0) {
            angle_deg += 360.0;
        }
        return angle_deg;
    }

    // ============================================================
    // @brief Compute the signed shortest angular difference between two angles.
    // ============================================================
    static double SignedShortestDeltaDeg(double from_deg, double to_deg) {
        const double normalized_from = NormalizeAngleDeg(from_deg);
        const double normalized_to = NormalizeAngleDeg(to_deg);
        return std::remainder(normalized_to - normalized_from, 360.0);
    }

    // ============================================================
    // @brief Compute an angle error with preferred rotation direction.
    // ============================================================
    static double ComputeDirectedAngleErrorDeg(double current_deg,
                                               double target_deg,
                                               bool prefer_positive_rotation,
                                               double threshold_deg) {
        const double shortest_error =
            SignedShortestDeltaDeg(current_deg, target_deg);
        if (std::abs(shortest_error) <= threshold_deg) {
            return shortest_error;
        }
        if (prefer_positive_rotation && shortest_error < 0.0) {
            return shortest_error + 360.0;
        }
        if (!prefer_positive_rotation && shortest_error > 0.0) {
            return shortest_error - 360.0;
        }
        return shortest_error;
    }

    // ============================================================
    // @brief Compute the angle difference for a drive-compatible swing target.
    // ============================================================
    static double ComputeDriveAngleDiffDeg(double target_swing_angle_deg,
                                           double swing_angle_deg,
                                           double threshold_deg) {
        const double target = std::abs(target_swing_angle_deg);
        double direction = 1.0;
        if (target < 0.0) {
            direction = -1.0;
        }
        const double diff =
            std::fmod(swing_angle_deg - target + 180.0, 360.0) - 180.0;
        if (std::abs(diff) <= threshold_deg) {
            return diff;
        }
        if (direction == 1.0 && diff < 0.0) {
            return diff + 360.0;
        }
        if (direction == -1.0 && diff > 0.0) {
            return diff - 360.0;
        }
        return diff;
    }

    // ============================================================
    // @brief Check whether the target angle is drive-compatible: 0 deg or 180 deg.
    // ============================================================
    static bool IsDriveCompatibleTargetDeg(double target_deg) {
        const double normalized_target = NormalizeAngleDeg(target_deg);
        return std::abs(normalized_target) <= 1e-3 ||
               std::abs(normalized_target - 180.0) <= 1e-3;
    }

    // ============================================================
    // @brief Compute the remaining swing angle from the change in azimuth.
    // ============================================================
    static double ComputeSwingAngleDiffAzimuthDeg(double start_deg,
                                                  double end_deg,
                                                  double target_angle_deg) {
        const double normalized_start = NormalizeAngleDeg(start_deg);
        const double normalized_end = NormalizeAngleDeg(end_deg);
        if (normalized_end < normalized_start) {
            return target_angle_deg -
                   std::abs(360.0 - normalized_start + normalized_end);
        }
        return target_angle_deg - std::abs(normalized_end - normalized_start);
    }

    // ============================================================
    // @brief Compute Euler angles from the acceleration vector.
    // ============================================================
    static void ComputeEulerAnglesDeg(double linear_acceleration_x,
                                      double linear_acceleration_y,
                                      double linear_acceleration_z,
                                      double& euler_x_deg,
                                      double& euler_y_deg) {
        euler_x_deg =
            std::atan2(-linear_acceleration_y, -linear_acceleration_z) *
            180.0 / 3.14159265358979323846;
        euler_y_deg =
            std::atan2(linear_acceleration_x,
                       std::sqrt(linear_acceleration_y *
                                     linear_acceleration_y +
                                 linear_acceleration_z *
                                     linear_acceleration_z)) *
            180.0 / 3.14159265358979323846;
    }

    // ============================================================
    // @brief Compute a drive-like PID-equivalent swing lever command.
    // ============================================================
    PidControlResult ComputeDriveLikePidControl(
        double swing_angle_diff_deg, double swing_velocity_median_rad_s,
        double& e_integration, double start_angle_diff_deg) const {
        PidControlResult result;

        double adjusted_ki = upper_body_pid_ki_;
        const double start_angle_diff_abs = std::abs(start_angle_diff_deg);
        if (start_angle_diff_abs <= 20.0) {
            adjusted_ki *= 180.0 / 20.0;
        } else if (start_angle_diff_abs <= 180.0) {
            adjusted_ki *= 180.0 / start_angle_diff_abs;
        } else {
            adjusted_ki *=
                (180.0 * 180.0 -
                 upper_body_pid_dec_target_angle_deg_ *
                     upper_body_pid_dec_target_angle_deg_) /
                (start_angle_diff_abs * start_angle_diff_abs -
                 upper_body_pid_dec_target_angle_deg_ *
                     upper_body_pid_dec_target_angle_deg_);
        }

        const double swing_velocity_median_deg_s =
            swing_velocity_median_rad_s * (180.0 / 3.14159265358979323846);
        result.p_control =
            upper_body_pid_kp_ * swing_angle_diff_deg + upper_body_pid_bias_;
        result.d_control =
            upper_body_pid_kd_ * swing_velocity_median_deg_s * (-1.0);

        if (std::abs(swing_angle_diff_deg) < upper_body_pid_threshold_deg_ &&
            std::abs(swing_velocity_median_rad_s) <
                upper_body_pid_threshold_imu_) {
            const double swing_angle_diff_i =
                std::max(swing_angle_diff_deg + 5.0, 1.0);
            e_integration += swing_angle_diff_i * 0.05;
        }

        result.i_control = adjusted_ki * e_integration;
        double swing_lever =
            result.p_control + result.d_control + result.i_control;
        swing_lever = std::clamp(swing_lever, -upper_body_pid_max_lever_abs_,
                                 upper_body_pid_max_lever_abs_);
        swing_lever = std::max(std::abs(swing_lever),
                               upper_body_pid_min_lever_abs_);
        result.lever = (-1.0) * swing_lever;
        return result;
    }

    // ============================================================
    // @brief Compute the correction lever used for center recovery.
    // ============================================================
    double ComputeRecoverLever(uint8_t state_swing,
                               uint8_t current_state_swing,
                               double swing_velocity_median_rad_s,
                               double previous_lever) const {
        constexpr double kDefaultLever = 0.3;
        constexpr double kAccelGain = 0.005;
        constexpr double kDeccelGain = 0.0;
        constexpr double kThresholdImu = 0.04;

        if (state_swing == 4U) {
            if (current_state_swing != 4U) {
                return -kDefaultLever;
            }
            if (std::abs(swing_velocity_median_rad_s) > kThresholdImu) {
                return previous_lever + kDeccelGain;
            }
            return previous_lever - kAccelGain;
        }

        if (state_swing == 5U) {
            if (current_state_swing != 5U) {
                return kDefaultLever;
            }
            if (std::abs(swing_velocity_median_rad_s) > kThresholdImu) {
                return previous_lever - kDeccelGain;
            }
            return previous_lever + kAccelGain;
        }

        return 0.0;
    }

    static double ComputeConstSwingLever() {
        return -0.5;
    }

    // ============================================================
    // @brief Publish a value to a Float64 debug topic.
    // ============================================================
    static void PublishFloat(rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr
                                 publisher,
                             double value) {
        if (!publisher) {
            return;
        }
        std_msgs::msg::Float64 msg;
        msg.data = value;
        publisher->publish(msg);
    }

    // ============================================================
    // @brief Publish debug values for control, IMU, and state transitions.
    // ============================================================
    void PublishDebugData(double swing_angle_diff_deg, double p_control,
                          double i_control, double d_control,
                          double swing_velocity_median_rad_s,
                          double e_integration, uint8_t state_swing,
                          uint8_t current_state_swing,
                          double center_pass_sw,
                          uint8_t finish_sw,
                          double swing_angle_diff_azimuth_deg,
                          double yaw_angle_deg, double euler_x_median_deg,
                          double euler_y_median_deg,
                          double linear_acceleration_magnitude,
                          double linear_acceleration_x_mean,
                          double linear_acceleration_y_mean,
                          double linear_acceleration_z_mean,
                          double angular_velocity_y_mean,
                          double linear_acceleration_magnitude_mean,
                          double target_swing_angle_deg) {
        if (state_swing_pub_) {
            std_msgs::msg::UInt8 msg;
            msg.data = state_swing;
            state_swing_pub_->publish(msg);
        }
        if (finish_sw_pub_) {
            std_msgs::msg::UInt8 msg;
            msg.data = finish_sw;
            finish_sw_pub_->publish(msg);
        }
        PublishFloat(center_pass_sw_pub_, center_pass_sw);
        PublishFloat(target_swing_angle_pub_, target_swing_angle_deg);
        PublishFloat(swing_angle_diff_azimuth_pub_,
                     swing_angle_diff_azimuth_deg);
        PublishFloat(yaw_angle_degrees_pub_, yaw_angle_deg);
        PublishFloat(linear_acceleration_magnitude_pub_,
                     linear_acceleration_magnitude);
        PublishFloat(linear_acceleration_x_mean_pub_,
                     linear_acceleration_x_mean);
        PublishFloat(linear_acceleration_y_mean_pub_,
                     linear_acceleration_y_mean);
        PublishFloat(linear_acceleration_z_mean_pub_,
                     linear_acceleration_z_mean);
        PublishFloat(angular_velocity_y_mean_pub_, angular_velocity_y_mean);
        PublishFloat(linear_acceleration_magnitude_mean_pub_,
                     linear_acceleration_magnitude_mean);
        PublishFloat(swing_angle_diff_pub_, swing_angle_diff_deg);
        PublishFloat(p_control_pub_, p_control);
        PublishFloat(i_control_pub_, i_control);
        PublishFloat(d_control_pub_, d_control);
        PublishFloat(swing_velocity_median_pub_, swing_velocity_median_rad_s);
        PublishFloat(euler_x_median_pub_, euler_x_median_deg);
        PublishFloat(euler_y_median_pub_, euler_y_median_deg);
        PublishFloat(e_integration_pub_, e_integration);

        if (current_state_swing_pub_) {
            std_msgs::msg::UInt8 msg;
            msg.data = current_state_swing;
            current_state_swing_pub_->publish(msg);
        }
    }

    // ============================================================
    // @brief Read the yaw angle derived from odometry.
    // ============================================================
    void ReadYawAngle(bool& has_yaw_angle, double& yaw_angle_deg) const {
        std::lock_guard<std::mutex> lock(yaw_angle_mutex_);
        has_yaw_angle = has_yaw_angle_;
        yaw_angle_deg = latest_yaw_angle_deg_;
    }

    // ============================================================
    // @brief Read the latest IMU linear acceleration values.
    // ============================================================
    void ReadImuLinearAcceleration(bool& has_imu_linear_acceleration,
                                   double& linear_acceleration_x,
                                   double& linear_acceleration_y,
                                   double& linear_acceleration_z) const {
        std::lock_guard<std::mutex> lock(imu_state_mutex_);
        has_imu_linear_acceleration = has_imu_linear_acceleration_;
        linear_acceleration_x = latest_linear_acceleration_x_;
        linear_acceleration_y = latest_linear_acceleration_y_;
        linear_acceleration_z = latest_linear_acceleration_z_;
    }

    // ============================================================
    // @brief Read the swing angular velocity derived from the IMU.
    // ============================================================
    void ReadSwingVelocity(bool& has_swing_velocity,
                           double& swing_velocity_rad_s) const {
        std::lock_guard<std::mutex> lock(imu_state_mutex_);
        has_swing_velocity = has_imu_linear_acceleration_;
        swing_velocity_rad_s = latest_swing_velocity_rad_s_;
    }

    // ============================================================
    // @brief Read the latest IMU Y-axis angular velocity.
    // ============================================================
    void ReadImuAngularVelocityY(double& angular_velocity_y) const {
        std::lock_guard<std::mutex> lock(imu_state_mutex_);
        angular_velocity_y = latest_angular_velocity_y_;
    }

    rclcpp::Node* node_{nullptr};
    std::shared_ptr<cd110r_controller::CD110RCommandState> command_state_;
    std::shared_ptr<cd110r_controller::CD110RFeedbackState> feedback_state_;

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;

    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr
        swing_angle_diff_pub_;
    rclcpp::Publisher<std_msgs::msg::UInt8>::SharedPtr state_swing_pub_;
    rclcpp::Publisher<std_msgs::msg::UInt8>::SharedPtr finish_sw_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr center_pass_sw_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr target_swing_angle_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr
        swing_angle_diff_azimuth_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr yaw_angle_degrees_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr
        linear_acceleration_magnitude_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr
        linear_acceleration_x_mean_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr
        linear_acceleration_y_mean_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr
        linear_acceleration_z_mean_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr
        angular_velocity_y_mean_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr
        linear_acceleration_magnitude_mean_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr p_control_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr i_control_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr d_control_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr
        swing_velocity_median_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr euler_x_median_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr euler_y_median_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr e_integration_pub_;
    rclcpp::Publisher<std_msgs::msg::UInt8>::SharedPtr current_state_swing_pub_;

    std::atomic<bool> goal_active_{false};

    mutable std::mutex yaw_angle_mutex_;
    bool has_yaw_angle_{false};
    double latest_yaw_angle_deg_{0.0};

    mutable std::mutex imu_state_mutex_;
    bool has_imu_linear_acceleration_{false};
    double latest_swing_velocity_rad_s_{0.0};
    double latest_angular_velocity_y_{0.0};
    double latest_linear_acceleration_x_{0.0};
    double latest_linear_acceleration_y_{0.0};
    double latest_linear_acceleration_z_{0.0};

    double upper_body_trajectory_timeout_sec_{90.0};
    double upper_body_switch_wait_timeout_sec_{3.0};
    double upper_body_switch_on_threshold_{0.5};
    double upper_body_switch_stable_sec_{1.0};
    double upper_body_reverse_toggle_threshold_deg_{90.0};
    double upper_body_center_align_timeout_sec_{12.0};
    std::string upper_body_odom_topic_;
    std::string upper_body_imu_topic_;

    double upper_body_pid_kp_{0.032};
    double upper_body_pid_ki_{0.005};
    double upper_body_pid_kd_{0.035};
    double upper_body_pid_threshold_deg_{40.0};
    double upper_body_pid_threshold_imu_{0.05};
    double upper_body_pid_dec_target_angle_deg_{45.0};
    double upper_body_pid_max_lever_abs_{1.0};
    double upper_body_pid_bias_{0.4};
    double upper_body_pid_min_lever_abs_{0.1};
    double upper_body_goal_tolerance_deg_{1.5};
    double upper_body_angle_diff_threshold_deg_{45.0};
    std::size_t upper_body_velocity_median_window_{5U};
    std::size_t upper_body_euler_median_window_{10U};
    std::size_t upper_body_mean_window_{5U};
    int upper_body_control_period_ms_{50};
};

}  // namespace cd110r_plugins

PLUGINLIB_EXPORT_CLASS(
    cd110r_plugins::Cd110UpperBodyTrajectoryHandler,
    vehicle_controller_interfaces::UpperBodyTrajectoryHandler)
