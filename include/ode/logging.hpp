/**
 * @file logging.hpp
 * @brief Standardized logging helpers for examples/tools/tests.
 */
#pragma once

#include <cstdio>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

namespace ode::log {

enum class Level {
  kInfo,
  kWarn,
  kError,
};

inline std::string_view ToString(Level level) {
  switch (level) {
    case Level::kInfo:
      return "info";
    case Level::kWarn:
      return "warn";
    case Level::kError:
      return "error";
  }
  return "unknown";
}

template <typename... Args>
inline std::string BuildMessage(Args&&... args) {
  std::ostringstream oss;
  (oss << ... << std::forward<Args>(args));
  return oss.str();
}

inline void Write(Level level, std::string_view message) {
  FILE* stream = (level == Level::kError) ? stderr : stdout;
  std::fprintf(stream, "[%s] %.*s\n",
               ToString(level).data(),
               static_cast<int>(message.size()),
               message.data());
}

template <typename... Args>
inline void Info(Args&&... args) {
  Write(Level::kInfo, BuildMessage(std::forward<Args>(args)...));
}

template <typename... Args>
inline void Warn(Args&&... args) {
  Write(Level::kWarn, BuildMessage(std::forward<Args>(args)...));
}

template <typename... Args>
inline void Error(Args&&... args) {
  Write(Level::kError, BuildMessage(std::forward<Args>(args)...));
}

}  // namespace ode::log
