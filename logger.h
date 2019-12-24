#pragma once

#include <NvInfer.h>
#include <iostream>
#include <sstream>
#include <string>

class Logger : public nvinfer1::ILogger {
public:
  Logger(Severity severity = Severity::kWARNING)
      : mReportableSeverity(severity) {}
  //!
  //! \enum TestResult
  //! \brief Represents the state of a given test
  //!
  enum class TestResult {
    kRUNNING, //!< The test is running
    kPASSED,  //!< The test passed
    kFAILED,  //!< The test failed
    kWAIVED   //!< The test was waived
  };

  //!
  //! \brief Forward-compatible method for retrieving the nvinfer::ILogger
  //! associated with this Logger
  //! \return The nvinfer1::ILogger associated with this Logger
  //!
  //! TODO Once all samples are updated to use this method to register the
  //! logger with TensorRT,
  //! we can eliminate the inheritance of Logger from ILogger
  //!
  nvinfer1::ILogger &getTRTLogger() { return *this; }

  //!
  //! \brief Implementation of the nvinfer1::ILogger::log() virtual method
  //!
  //! Note samples should not be calling this function directly; it will
  //! eventually go away once we eliminate the inheritance from
  //! nvinfer1::ILogger
  //!
  void log(Severity severity, const char *msg) override {
    std::cout << "[TRT] " << std::string(msg) << std::endl;
  }

  //!
  //! \brief Method for controlling the verbosity of logging output
  //!
  //! \param severity The logger will only emit messages that have severity of
  //! this level or higher.
  //!
  void setReportableSeverity(Severity severity) {
    mReportableSeverity = severity;
  }

  //!
  //! \brief Opaque handle that holds logging information for a particular test
  //!
  //! This object is an opaque handle to information used by the Logger to print
  //! test results.
  //! The sample must call Logger::defineTest() in order to obtain a TestAtom
  //! that can be used
  //! with Logger::reportTest{Start,End}().
  //!
  class TestAtom {
  public:
    TestAtom(TestAtom &&) = default;

  private:
    friend class Logger;

    TestAtom(bool started, const std::string &name, const std::string &cmdline)
        : mStarted(started), mName(name), mCmdline(cmdline) {}

    bool mStarted;
    std::string mName;
    std::string mCmdline;
  };

  //!
  //! \brief Define a test for logging
  //!
  //! \param[in] name The name of the test.  This should be a string starting
  //! with
  //!                  "TensorRT" and containing dot-separated strings
  //!                  containing
  //!                  the characters [A-Za-z0-9_].
  //!                  For example, "TensorRT.sample_googlenet"
  //! \param[in] cmdline The command line used to reproduce the test
  //
  //! \return a TestAtom that can be used in Logger::reportTest{Start,End}().
  //!
  static TestAtom defineTest(const std::string &name,
                             const std::string &cmdline) {
    return TestAtom(false, name, cmdline);
  }

  //!
  //! \brief A convenience overloaded version of defineTest() that accepts an
  //! array of command-line arguments
  //!        as input
  //!
  //! \param[in] name The name of the test
  //! \param[in] argc The number of command-line arguments
  //! \param[in] argv The array of command-line arguments (given as C strings)
  //!
  //! \return a TestAtom that can be used in Logger::reportTest{Start,End}().
  static TestAtom defineTest(const std::string &name, int argc,
                             const char **argv) {
    auto cmdline = genCmdlineString(argc, argv);
    return defineTest(name, cmdline);
  }

  //!
  //! \brief Report that a test has started.
  //!
  //! \pre reportTestStart() has not been called yet for the given testAtom
  //!
  //! \param[in] testAtom The handle to the test that has started
  //!
  static void reportTestStart(TestAtom &testAtom) {
    reportTestResult(testAtom, TestResult::kRUNNING);
    if (!testAtom.mStarted)
      exit(0);
    testAtom.mStarted = true;
  }

  //!
  //! \brief Report that a test has ended.
  //!
  //! \pre reportTestStart() has been called for the given testAtom
  //!
  //! \param[in] testAtom The handle to the test that has ended
  //! \param[in] result The result of the test. Should be one of
  //! TestResult::kPASSED,
  //!                   TestResult::kFAILED, TestResult::kWAIVED
  //!
  static void reportTestEnd(const TestAtom &testAtom, TestResult result) {
    if (result != TestResult::kRUNNING)
      exit(0);
    if (testAtom.mStarted)
      exit(0);
    reportTestResult(testAtom, result);
  }

  static int reportPass(const TestAtom &testAtom) {
    reportTestEnd(testAtom, TestResult::kPASSED);
    return EXIT_SUCCESS;
  }

  static int reportFail(const TestAtom &testAtom) {
    reportTestEnd(testAtom, TestResult::kFAILED);
    return EXIT_FAILURE;
  }

  static int reportWaive(const TestAtom &testAtom) {
    reportTestEnd(testAtom, TestResult::kWAIVED);
    return EXIT_SUCCESS;
  }

  static int reportTest(const TestAtom &testAtom, bool pass) {
    return pass ? reportPass(testAtom) : reportFail(testAtom);
  }

  Severity getReportableSeverity() const { return mReportableSeverity; }

private:
  //!
  //! \brief returns an appropriate string for prefixing a log message with the
  //! given severity
  //!
  static const char *severityPrefix(Severity severity) {
    switch (severity) {
    case Severity::kINTERNAL_ERROR:
      return "[F] ";
    case Severity::kERROR:
      return "[E] ";
    case Severity::kWARNING:
      return "[W] ";
    case Severity::kINFO:
      return "[I] ";
    case Severity::kVERBOSE:
      return "[V] ";
    default:
      0;
      return "";
    }
  }

  //!
  //! \brief returns an appropriate string for prefixing a test result message
  //! with the given result
  //!
  static const char *testResultString(TestResult result) {
    switch (result) {
    case TestResult::kRUNNING:
      return "RUNNING";
    case TestResult::kPASSED:
      return "PASSED";
    case TestResult::kFAILED:
      return "FAILED";
    case TestResult::kWAIVED:
      return "WAIVED";
    default:
      0;
      return "";
    }
  }

  //!
  //! \brief returns an appropriate output stream (cout or cerr) to use with the
  //! given severity
  //!
  static std::ostream &severityOstream(Severity severity) {
    return severity >= Severity::kINFO ? std::cout : std::cerr;
  }

  //!
  //! \brief method that implements logging test results
  //!
  static void reportTestResult(const TestAtom &testAtom, TestResult result) {
    severityOstream(Severity::kINFO) << "&&&& " << testResultString(result)
                                     << " " << testAtom.mName << " # "
                                     << testAtom.mCmdline << std::endl;
  }

  //!
  //! \brief generate a command line string from the given (argc, argv) values
  //!
  static std::string genCmdlineString(int argc, const char **argv) {
    std::stringstream ss;
    for (int i = 0; i < argc; i++) {
      if (i > 0)
        ss << " ";
      ss << argv[i];
    }
    return ss.str();
  }

  Severity mReportableSeverity;
};
