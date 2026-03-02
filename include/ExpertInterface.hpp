#pragma once

#include <memory>
#include <string>
#include <vector>

namespace rmoe {

using ConfidenceScore = float;

enum class ExpertTarget {
    Reasoning,
    Clinical
};

struct FeedbackTensor {
    std::string request_type;
    std::string payload;
};

struct DiagnosticData {
    ConfidenceScore sc {0.0F};
    std::string analysis;
    FeedbackTensor feedback {};
    // DDx ensemble probability samples used by ARLL to compute Sc = 1 - sigma^2
    std::vector<float> ddx_probabilities;
};

struct UncertaintyMetrics {
    ConfidenceScore confidence {0.0F};
    float uncertainty {1.0F};
    float predictive_entropy {0.0F};
    float ddx_variance {0.0F}; // sigma^2 in Sc = 1 - sigma^2 (ensemble DDx variance)
};

struct IterationTrace {
    int iteration {1};
    std::string perception_summary;
    std::string reasoning_summary;
    std::string decision;
    UncertaintyMetrics metrics {};
};

struct RunSummary {
    bool success {false};
    bool escalated_to_human {false};
    int iterations_executed {0};
    std::string final_report_json;
    std::vector<IterationTrace> trace;
};

struct ModelSettings {
    std::string vision_projection_model {"models/vision_proj.gguf"};
    std::string vision_text_model {"models/vision_text.gguf"};
    std::string reasoning_model {"models/reasoning_expert.gguf"};
    std::string clinical_model {"models/clinical_expert.gguf"};
};

class IClinicalExpert {
public:
    virtual ~IClinicalExpert() = default;
    virtual DiagnosticData execute(const std::string& input_data) = 0;
    [[nodiscard]] virtual std::string get_expert_name() const = 0;
};

class IPerceptionExpert : public IClinicalExpert {
public:
    ~IPerceptionExpert() override = default;
};

class IReasoningExpert : public IClinicalExpert {
public:
    ~IReasoningExpert() override = default;
};

class IReportingExpert : public IClinicalExpert {
public:
    ~IReportingExpert() override = default;
};

enum class WannaState {
    ProceedToReport,
    RequestHighResCrop,
    RequestAlternateView,
    EscalateToHuman
};

struct WannaDecision {
    WannaState state {WannaState::ProceedToReport};
    int iteration {1};
    FeedbackTensor feedback {};
};

class WannaStateMachine {
public:
    WannaStateMachine(int hard_limit_iterations, ConfidenceScore threshold);

    [[nodiscard]] WannaDecision evaluate(const DiagnosticData& reasoning_result, int iteration) const;
    [[nodiscard]] int hard_limit_iterations() const noexcept;
    [[nodiscard]] ConfidenceScore threshold() const noexcept;

private:
    int hard_limit_iterations_;
    ConfidenceScore threshold_;
};

class ExpertSwapper {
public:
    ExpertSwapper() = default;
    ~ExpertSwapper();

    void unload_current_expert();
    bool load_expert_model(const std::string& model_path, int n_ctx = 2048);
    [[nodiscard]] std::string infer_text(const std::string& prompt, int max_new_tokens = 32) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class DiagnosticEngine {
public:
    explicit DiagnosticEngine(WannaStateMachine state_machine, ModelSettings settings = {});
    RunSummary run_diagnostics(const std::string& patient_input);

private:
    static void escalate_to_human(const DiagnosticData& diagnostic_data);

    WannaStateMachine state_machine_;
    ModelSettings settings_;
    ExpertSwapper swapper_;
};

class MrTom {
public:
    explicit MrTom(WannaStateMachine state_machine);

    void set_vision_model(const std::string& projection_model_path, const std::string& text_model_path);
    void set_reasoning_model(const std::string& reasoning_model_path);
    void set_clinical_model(const std::string& clinical_model_path);
    void configure_gate(int max_iterations, ConfidenceScore threshold);
    bool load_settings(const std::string& settings_json_path);

    RunSummary process_patient_case(const std::string& patient_input) const;
    std::string ask_expert(const std::string& question, ExpertTarget target = ExpertTarget::Reasoning) const;

private:
    WannaStateMachine state_machine_;
    ModelSettings settings_;
};

void initialize_backend();
void shutdown_backend();

} // namespace rmoe
