#pragma once

#include <string>
#include <vector>
#include <map>
#include <set>
#include <functional>
#include <iostream>
#include <memory>
#include <typeindex>
#include <iomanip>

// Configuration for test filtering
struct TestConfig {
    std::string category_filter; // "avx", "avx2", "avx512" or "" (all)
    std::string class_filter;    // match for CLASS_TYPE (e.g. "cmp", "bitwise")
    std::string op_filter;       // match for op name
    std::string type_filter;     // partial match for type signature
    bool help_mode = false;      // If true, collect info and print help
    bool verbose = false;        // Print detailed output even on success
};

// Base interface for a test case
class ITest {
public:
    virtual ~ITest() = default;
    virtual void Run(const TestConfig& config) = 0;
    virtual std::string GetCategory() const = 0;
    virtual std::string GetClassType() const = 0;
    virtual std::vector<std::string> GetOpNames() const = 0;
};

// Registry for all tests
class TestRegistry {
public:
    static TestRegistry& Instance() {
        static TestRegistry instance;
        return instance;
    }

    void Register(std::unique_ptr<ITest> test) {
        tests_.push_back(std::move(test));
    }

    const std::vector<std::unique_ptr<ITest>>& GetTests() const {
        return tests_;
    }

    void PrintHelp(const TestConfig& config, const char* prog_name) {
        // Collect metadata
        std::set<std::string> categories;
        std::map<std::string, std::set<std::string>> category_classes;
        std::map<std::pair<std::string, std::string>, std::set<std::string>> class_ops;

        for (const auto& test : tests_) {
            std::string cat = test->GetCategory();
            std::string cls = test->GetClassType();
            categories.insert(cat);
            category_classes[cat].insert(cls);
            for (const auto& op : test->GetOpNames()) {
                class_ops[{cat, cls}].insert(op);
            }
        }

        std::cout << "Usage: " << prog_name << " [options]\n\n";
        
        if (config.category_filter.empty()) {
            std::cout << "Available Categories (use --avx<version> --help to see details):\n";
            for (const auto& cat : categories) {
                std::cout << "  --avx" << (cat.substr(0, 3) == "avx" ? cat.substr(3) : cat) << "\n";
            }
        } else {
            std::string cat = config.category_filter;
            if (categories.find(cat) == categories.end()) {
                std::cout << "Error: Category '" << cat << "' not found.\n";
                return;
            }

            if (config.class_filter.empty()) {
                std::cout << "Available Classes in " << cat << " (use --class=<name> --help to see operations):\n";
                for (const auto& cls : category_classes[cat]) {
                    std::cout << "  --class=" << cls << "\n";
                }
            } else {
                std::string cls = config.class_filter;
                if (category_classes[cat].find(cls) == category_classes[cat].end()) {
                    std::cout << "Error: Class '" << cls << "' not found in category '" << cat << "'.\n";
                    return;
                }

                std::cout << "Available Operations in " << cat << "::" << cls << ":\n";
                for (const auto& op : class_ops[{cat, cls}]) {
                    std::cout << "  --op=" << op << "\n";
                }
            }
        }
        
        std::cout << "\nGeneral Options:\n";
        std::cout << "  --op=<name>      Filter by operation name (partial match)\n";
        std::cout << "  --type=<sig>     Filter by type signature (partial match)\n";
        std::cout << "  --help           Show this help message\n";
    }

private:
    std::vector<std::unique_ptr<ITest>> tests_;
};

// Helper for auto-registration
template <typename T>
struct AutoRegister {
    AutoRegister() {
        TestRegistry::Instance().Register(std::make_unique<T>());
    }
};

// Colors for output
namespace Color {
    const char* Reset   = "\033[0m";
    const char* Red     = "\033[31m";
    const char* Green   = "\033[32m";
    const char* Yellow  = "\033[33m";
    const char* Blue    = "\033[34m";
    const char* Cyan    = "\033[36m";
}
