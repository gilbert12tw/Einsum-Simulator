
/**
 * EinsumSimulator - A CUDA-Q NVQIR simulator that intercepts quantum gates
 * and builds Einsum expressions for tensor network contraction.
 *
 * This simulator does not perform actual quantum simulation, but instead
 * records the sequence of quantum operations to generate Einsum notation.
 */

#include "nvqir/CircuitSimulator.h"
#include <map>
#include <string>
#include <vector>
#include <complex>
#include <sstream>
#include <iostream>
#include <memory>
#include <cstring>

// ============================================================================
// Sidecar Buffer - Thread-local storage for Python access via ctypes
// ============================================================================
static thread_local std::string g_einsum_buffer;

extern "C" {
    /// Get the length of the current einsum buffer
    int get_einsum_length() {
        return static_cast<int>(g_einsum_buffer.length());
    }

    /// Copy the einsum data to the provided buffer
    void get_einsum_data(char* buffer) {
        if (buffer && !g_einsum_buffer.empty()) {
            std::memcpy(buffer, g_einsum_buffer.c_str(), g_einsum_buffer.length());
            buffer[g_einsum_buffer.length()] = '\0';
        }
    }

    /// Clear the einsum buffer
    void clear_einsum_buffer() {
        g_einsum_buffer.clear();
    }
}

namespace {

/**
 * EinsumBuilder - Intercepts quantum gates and builds Einsum expressions.
 *
 * Inherits from CircuitSimulatorBase<double> to use double precision.
 */
class EinsumBuilder : public nvqir::CircuitSimulatorBase<double> {
private:
    // Track qubit indices - maps qubit ID to current tensor index
    std::map<std::size_t, std::size_t> qubitIndices;

    // Global index counter
    std::size_t nextIndex = 0;

    // Number of qubits allocated
    std::size_t numQubits = 0;

    // Accumulated gate operations for Einsum generation
    struct GateRecord {
        std::string name;
        std::vector<double> params;
        std::vector<std::size_t> controls;
        std::vector<std::size_t> targets;
        std::vector<std::size_t> inputIndices;
        std::vector<std::size_t> outputIndices;
        // Store the gate matrix
        std::vector<std::complex<double>> matrix;
    };
    std::vector<GateRecord> gateHistory;

    // Initial state records
    struct InitialState {
        std::size_t qubitId;
        std::size_t outputIndex;
    };
    std::vector<InitialState> initialStates;

public:
    EinsumBuilder() = default;
    ~EinsumBuilder() override = default;

    /// Return the name of this simulator
    std::string name() const override { return "einsum"; }

    /// Clone this simulator (required for thread-local instances)
    NVQIR_SIMULATOR_CLONE_IMPL(EinsumBuilder)

    /// Determine if this is single precision (we use double)
    bool isSinglePrecision() const override { return false; }

    /// Add a qubit to the state representation
    void addQubitToState() override {
        std::size_t qubitId = numQubits++;
        std::size_t idx = nextIndex++;
        qubitIndices[qubitId] = idx;

        // Record initial state
        initialStates.push_back({qubitId, idx});

        std::cout << "[Einsum] Allocated qubit " << qubitId
                  << " with initial index " << idx << std::endl;
    }

    /// Handle state deallocation
    void deallocateStateImpl() override {
        // Export to sidecar buffer before clearing
        exportToSidecar();

        qubitIndices.clear();
        gateHistory.clear();
        initialStates.clear();
        nextIndex = 0;
        numQubits = 0;
        std::cout << "[Einsum] State deallocated" << std::endl;
    }

    /// Reset the state to |0...0>
    void setToZeroState() override {
        gateHistory.clear();
        initialStates.clear();
        nextIndex = 0;

        // Re-initialize all qubits
        for (std::size_t q = 0; q < numQubits; ++q) {
            std::size_t idx = nextIndex++;
            qubitIndices[q] = idx;
            initialStates.push_back({q, idx});
        }
        std::cout << "[Einsum] State reset to |0>" << std::endl;
    }

    /// Core method: intercept gate operations and record for Einsum
    void applyGate(const GateApplicationTask& task) override {
        GateRecord record;
        record.name = task.operationName;
        record.params = std::vector<double>(task.parameters.begin(), task.parameters.end());
        record.controls = task.controls;
        record.targets = task.targets;

        // Store the gate matrix
        for (const auto& elem : task.matrix) {
            record.matrix.push_back(elem);
        }

        // Collect current indices for all involved qubits (input indices)
        for (auto q : task.controls) {
            record.inputIndices.push_back(qubitIndices[q]);
        }
        for (auto q : task.targets) {
            record.inputIndices.push_back(qubitIndices[q]);
        }

        // Allocate new output indices
        for (auto q : task.controls) {
            std::size_t newIdx = nextIndex++;
            record.outputIndices.push_back(newIdx);
            qubitIndices[q] = newIdx;
        }
        for (auto q : task.targets) {
            std::size_t newIdx = nextIndex++;
            record.outputIndices.push_back(newIdx);
            qubitIndices[q] = newIdx;
        }

        gateHistory.push_back(record);

        // Log the gate operation
        std::cout << "[Einsum] Gate: " << task.operationName;
        if (!task.parameters.empty()) {
            std::cout << "(";
            for (size_t i = 0; i < task.parameters.size(); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << task.parameters[i];
            }
            std::cout << ")";
        }
        std::cout << " | in: [";
        for (size_t i = 0; i < record.inputIndices.size(); ++i) {
            if (i > 0) std::cout << ",";
            std::cout << record.inputIndices[i];
        }
        std::cout << "] -> out: [";
        for (size_t i = 0; i < record.outputIndices.size(); ++i) {
            if (i > 0) std::cout << ",";
            std::cout << record.outputIndices[i];
        }
        std::cout << "]" << std::endl;
    }

    /// Perform qubit measurement
    bool measureQubit(const std::size_t qubitIdx) override {
        std::cout << "[Einsum] Measure qubit " << qubitIdx
                  << " (index: " << qubitIndices[qubitIdx] << ")" << std::endl;
        return false;
    }

    /// Reset a single qubit to |0>
    void resetQubit(const std::size_t qubitIdx) override {
        qubitIndices[qubitIdx] = nextIndex++;
        std::cout << "[Einsum] Reset qubit " << qubitIdx << std::endl;
    }

    /// Sample the state - triggers export to sidecar
    cudaq::ExecutionResult sample(const std::vector<std::size_t>& qubits,
                                   const int shots) override {
        std::cout << "[Einsum] Sample requested on " << qubits.size()
                  << " qubits for " << shots << " shots" << std::endl;

        // Export to sidecar buffer
        exportToSidecar();

        // Return dummy result
        std::string bitstring(qubits.size(), '0');
        cudaq::ExecutionResult result;
        result.appendResult(bitstring, shots);
        return result;
    }

    /// Export circuit data to the sidecar buffer (JSON format)
    void exportToSidecar() {
        std::stringstream ss;
        ss << "{\n";

        // Metadata
        ss << "  \"numQubits\": " << numQubits << ",\n";
        ss << "  \"numGates\": " << gateHistory.size() << ",\n";
        ss << "  \"maxIndex\": " << nextIndex << ",\n";

        // Initial states
        ss << "  \"initialStates\": [\n";
        for (size_t i = 0; i < initialStates.size(); ++i) {
            ss << "    {\"qubitId\": " << initialStates[i].qubitId
               << ", \"index\": " << initialStates[i].outputIndex << "}";
            if (i < initialStates.size() - 1) ss << ",";
            ss << "\n";
        }
        ss << "  ],\n";

        // Gates
        ss << "  \"gates\": [\n";
        for (size_t i = 0; i < gateHistory.size(); ++i) {
            const auto& g = gateHistory[i];
            ss << "    {\n";
            ss << "      \"name\": \"" << g.name << "\",\n";

            // Parameters
            ss << "      \"params\": [";
            for (size_t j = 0; j < g.params.size(); ++j) {
                if (j > 0) ss << ", ";
                ss << g.params[j];
            }
            ss << "],\n";

            // Controls
            ss << "      \"controls\": [";
            for (size_t j = 0; j < g.controls.size(); ++j) {
                if (j > 0) ss << ", ";
                ss << g.controls[j];
            }
            ss << "],\n";

            // Targets
            ss << "      \"targets\": [";
            for (size_t j = 0; j < g.targets.size(); ++j) {
                if (j > 0) ss << ", ";
                ss << g.targets[j];
            }
            ss << "],\n";

            // Input indices
            ss << "      \"inputIndices\": [";
            for (size_t j = 0; j < g.inputIndices.size(); ++j) {
                if (j > 0) ss << ", ";
                ss << g.inputIndices[j];
            }
            ss << "],\n";

            // Output indices
            ss << "      \"outputIndices\": [";
            for (size_t j = 0; j < g.outputIndices.size(); ++j) {
                if (j > 0) ss << ", ";
                ss << g.outputIndices[j];
            }
            ss << "],\n";

            // Matrix (as flat array of [real, imag] pairs)
            ss << "      \"matrix\": [";
            for (size_t j = 0; j < g.matrix.size(); ++j) {
                if (j > 0) ss << ", ";
                ss << "[" << g.matrix[j].real() << ", " << g.matrix[j].imag() << "]";
            }
            ss << "]\n";

            ss << "    }";
            if (i < gateHistory.size() - 1) ss << ",";
            ss << "\n";
        }
        ss << "  ],\n";

        // Output indices (final qubit indices)
        ss << "  \"outputIndices\": [";
        for (size_t i = 0; i < numQubits; ++i) {
            if (i > 0) ss << ", ";
            ss << qubitIndices[i];
        }
        ss << "]\n";

        ss << "}\n";

        // Store in sidecar buffer
        g_einsum_buffer = ss.str();
        std::cout << "[Einsum] Exported " << g_einsum_buffer.length()
                  << " bytes to sidecar buffer" << std::endl;
    }

    /// Observe a spin operator (not implemented)
    cudaq::observe_result observe(const cudaq::spin_op& op) override {
        throw std::runtime_error("EinsumBuilder does not support observe()");
    }

    /// Set noise model (not supported)
    void setNoiseModel(cudaq::noise_model& noise) override {
        throw std::runtime_error("EinsumBuilder does not support noise models");
    }

    /// Create state from data (not supported)
    std::unique_ptr<cudaq::SimulationState>
    createStateFromData(const cudaq::state_data& data) override {
        throw std::runtime_error("EinsumBuilder does not support state initialization from data");
    }
};

} // anonymous namespace

// Register this simulator with NVQIR
NVQIR_REGISTER_SIMULATOR(EinsumBuilder, einsum)
