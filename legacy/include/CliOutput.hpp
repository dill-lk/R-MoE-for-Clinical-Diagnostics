#pragma once

// ═══════════════════════════════════════════════════════════════════════════
//  CliOutput.hpp — R-MoE  Beautiful CLI output helpers
//  All functions write to std::cout using ANSI colour codes.
//  Internal/debug messages (model load/unload) live on std::cerr.
// ═══════════════════════════════════════════════════════════════════════════

#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace rmoe::cli {

// ─── Display width and truncation limits ────────────────────────────────────
static constexpr int         kWidth               = 72;
static constexpr std::size_t kMaxAbstainReasonLen = 120U;
static constexpr std::size_t kMaxNarrativeLen     = 200U;

// ─── ANSI codes ─────────────────────────────────────────────────────────────
static constexpr const char* kReset   = "\033[0m";
static constexpr const char* kBold    = "\033[1m";
static constexpr const char* kDim     = "\033[2m";
static constexpr const char* kCyan    = "\033[36m";
static constexpr const char* kGreen   = "\033[32m";
static constexpr const char* kYellow  = "\033[33m";
static constexpr const char* kRed     = "\033[31m";
static constexpr const char* kBlue    = "\033[34m";
static constexpr const char* kWhite   = "\033[97m";

// ─── Utilities ──────────────────────────────────────────────────────────────
inline std::string path_basename(const std::string& p) {
    const std::size_t pos = p.find_last_of("/\\");
    return (pos == std::string::npos) ? p : p.substr(pos + 1U);
}

inline void print_rule(char c = '=') {
    std::cout << kCyan << kDim
              << "  " << std::string(static_cast<std::size_t>(kWidth), c)
              << kReset << '\n';
}

inline void print_dash_rule() { print_rule('-'); }

inline void print_section(const std::string& title) {
    print_rule('=');
    const int pad = (kWidth - static_cast<int>(title.size())) / 2;
    std::cout << kCyan << kBold
              << std::string(static_cast<std::size_t>(pad > 0 ? pad + 2 : 2), ' ')
              << title << kReset << '\n';
    print_rule('=');
}

// ─── Top-level banner ────────────────────────────────────────────────────────
inline void print_banner() {
    std::cout
        << '\n'
        << kCyan << kBold
        << "  ========================================================================\n"
        << "    R-MoE  |  Recursive Multi-Agent Mixture-of-Experts  Clinical Engine\n"
        << "    Autonomous Medical Diagnostics  (llama.cpp backend)  [Research Build]\n"
        << "  ========================================================================\n"
        << kReset << '\n';
}

// ─── Patient / gate info ─────────────────────────────────────────────────────
inline void print_input_info(const std::string& image_path,
                              float gate_threshold, int max_iter) {
    std::cout
        << kWhite  << "  Patient Input   : " << kCyan << image_path << kReset << '\n'
        << kWhite  << "  Confidence Gate : " << kCyan
        << "Sc >= " << std::fixed << std::setprecision(2) << gate_threshold
        << kReset  << kWhite << "  |  Max Iterations : " << kCyan << max_iter
        << kReset  << '\n' << '\n';
    print_rule();
    std::cout << '\n';
}

// ─── Iteration header ────────────────────────────────────────────────────────
inline void print_iteration_header(int iter, int max_iter) {
    std::cout
        << '\n'
        << kYellow << kBold
        << "  ITERATION  " << iter << " / " << max_iter
        << kReset << '\n'
        << kYellow << kDim
        << "  " << std::string(static_cast<std::size_t>(kWidth), '-')
        << kReset << '\n';
}

// ─── Phase 1 – MPE ───────────────────────────────────────────────────────────
inline void print_mpe_status(const std::string& proj_path,
                              const std::string& text_path) {
    std::cout
        << '\n'
        << kBlue << kBold << "  [Phase 1]  " << kReset
        << kBold  << "MPE  Multi-Modal Perception Engine"
        << kReset << kDim << "  [Qwen2-VL-72B]\n" << kReset
        << kDim   << "             Projection : " << kReset << path_basename(proj_path) << '\n'
        << kDim   << "             Encoder    : " << kReset << path_basename(text_path) << '\n'
        << kGreen << "             Status     : OK  -  visual evidence extracted\n"
        << kReset;
}

// ─── Phase 2 – ARLL ──────────────────────────────────────────────────────────
inline void print_arll_result(float sc, float sigma2, float entropy,
                               bool gate_passed,
                               const std::string& request = "",
                               const std::string& payload = "") {
    std::cout
        << '\n'
        << kBlue << kBold << "  [Phase 2]  " << kReset
        << kBold  << "ARLL  Agentic Reasoning & Logic Layer"
        << kReset << kDim << "  [DeepSeek-R1]\n" << kReset
        << std::fixed << std::setprecision(4)
        << kDim    << "             sigma^2 = " << kReset << kCyan << sigma2
        << kDim    << "   Sc = "               << kReset << kCyan << sc
        << kDim    << "   H = "                << kReset << kCyan << entropy
        << kReset  << '\n';

    if (gate_passed) {
        std::cout << kGreen << kBold
                  << "             Gate    : PASS  (Sc >= 0.90)  ->  Proceed to CSR\n"
                  << kReset;
    } else {
        std::cout << kYellow << kBold
                  << "             Gate    : FAIL  (Sc < 0.90)   ->  #wanna# triggered\n"
                  << kReset;
        if (!request.empty()) {
            std::cout
                << kYellow << "             #wanna# : " << kReset << request << '\n'
                << kDim    << "             Payload : " << kReset << payload  << '\n';
        }
    }
}

// ─── Phase 3 – CSR ───────────────────────────────────────────────────────────
inline void print_csr_status(const std::string& model_path) {
    std::cout
        << '\n'
        << kBlue  << kBold << "  [Phase 3]  " << kReset
        << kBold  << "CSR  Clinical Synthesis & Reporting"
        << kReset << kDim << "   [Llama-3-Medius]\n" << kReset
        << kDim   << "             Model   : " << kReset << path_basename(model_path) << '\n'
        << kGreen << "             ICD-11 / SNOMED CT coding applied\n"
                  << "             Risk stratification (TIRADS / BI-RADS) computed\n"
                  << "             Status  : Report generated\n"
        << kReset;
}

// ─── Abstention ──────────────────────────────────────────────────────────────
inline void print_abstain(const std::string& reason) {
    const std::string display = reason.size() > kMaxAbstainReasonLen
        ? reason.substr(0U, kMaxAbstainReasonLen) + "..."
        : reason;
    std::cout
        << '\n'
        << kRed << kBold << "  [ABSTAIN]  Escalating to Human Radiologist\n" << kReset
        << kRed << kDim  << "             Reason  : " << display << kReset << '\n';
}

// ─── Key-value row helper (used in report display) ───────────────────────────
inline void print_kv(const std::string& key, const std::string& value,
                     const char* value_color = nullptr) {
    std::cout << kWhite << "  " << std::left << std::setw(18) << key
              << ": " << kReset;
    if (value_color) { std::cout << value_color; }
    std::cout << value << kReset << '\n';
}

} // namespace rmoe::cli
