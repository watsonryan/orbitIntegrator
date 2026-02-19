/**
 * @file logging.hpp
 * @brief spdlog-backed logging helpers for examples/tools/tests.
 */
#pragma once

#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

#include <spdlog/logger.h>
#include <spdlog/spdlog.h>

namespace ode::log {

inline std::shared_ptr<spdlog::logger> Logger() {
  return spdlog::default_logger();
}

template <typename... Args>
inline std::string BuildMessage(Args&&... args) {
  std::ostringstream oss;
  (oss << ... << std::forward<Args>(args));
  return oss.str();
}

template <typename... Args>
inline void Info(Args&&... args) {
  Logger()->info("{}", BuildMessage(std::forward<Args>(args)...));
}

template <typename... Args>
inline void Warn(Args&&... args) {
  Logger()->warn("{}", BuildMessage(std::forward<Args>(args)...));
}

template <typename... Args>
inline void Error(Args&&... args) {
  Logger()->error("{}", BuildMessage(std::forward<Args>(args)...));
}

}  // namespace ode::log
